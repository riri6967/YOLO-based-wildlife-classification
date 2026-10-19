"""Evaluate every model on the camera-site-disjoint test split with identical code.

Test sets: "boxed" = box-annotated test split (mAP, box metrics); "clean" = every test
image (boxed, empty, independent image-level) that the YOLOv4 reference never saw in
training (image-level, condition, threshold and ecology metrics; fair to all models).

Per model (results/eval/<model>/):
  metrics.json        COCO mAP, box- and image-level macro P/R/F1, empty filtering,
                      counts, richness and occupancy checks, speed and energy
  per_species.csv     AP50, box-level and image-level P/R/F1 per species, n_train
  per_image.csv       one row per test image (input for wildsim calibration)
  conditions.csv      recall and false-positive rate by night/day, season,
                      vegetation density and visibility
  threshold_sweep.csv metrics across confidence thresholds
  confusion.csv       image-level confusion matrix (true x predicted label)
Across models: results/eval/comparison.csv and figures.

    python -m nkpipe.evaluate
"""

import contextlib
import io
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import cli, load_classes, log, read_json, write_json  # noqa: E402
from . import preds as P  # noqa: E402

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# --------------------------------------------------------------------------- box level

def iou_xywh(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def box_counts(gt_by_uid, dets, thr, iou_thr, k):
    """Greedy class-aware matching; returns tp, fp, fn arrays per class."""
    tp, fp, fn = np.zeros(k), np.zeros(k), np.zeros(k)
    for uid in set(gt_by_uid) | set(dets):
        g = gt_by_uid.get(uid, [])
        d = sorted([x for x in dets.get(uid, []) if x[5] >= thr], key=lambda x: -x[5])
        used = np.zeros(len(g), bool)
        for x in d:
            c = x[4]
            cand = [j for j, gg in enumerate(g) if gg[4] == c and not used[j]]
            if cand:
                ious = iou_xywh(x[:4], [g[j][:4] for j in cand])[0]
                j = int(np.argmax(ious))
                if ious[j] >= iou_thr:
                    used[cand[j]] = True
                    tp[c] += 1
                    continue
            fp[c] += 1
        for j, gg in enumerate(g):
            if not used[j]:
                fn[gg[4]] += 1
    return tp, fp, fn


def prf(tp, fp, fn):
    p = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan, dtype=float), where=(tp + fn) > 0)
    f = np.divide(2 * p * r, p + r, out=np.full_like(tp, np.nan, dtype=float), where=(p + r) > 0)
    return p, r, f


def coco_map(paths, dets, k):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(paths.coco_test))
    uid2id = {im["uid"]: im["id"] for im in gt.dataset["images"]}
    res = [{"image_id": uid2id[u], "category_id": int(x[4]) + 1, "bbox": [float(v) for v in x[:4]],
            "score": float(x[5])} for u, ds in dets.items() if u in uid2id for x in ds]
    if not res or not gt.dataset["annotations"]:
        return {"mAP50_95": np.nan, "mAP50": np.nan}, np.full(k, np.nan)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
        E = COCOeval(gt, dt, "bbox")
        E.params.maxDets = [1, 10, 100]
        E.evaluate()
        E.accumulate()
        E.summarize()
    prec = E.eval["precision"]  # [T, R, K, A, M]
    ap50 = np.full(k, np.nan)
    for ki, cid in enumerate(E.params.catIds):
        p = prec[0, :, ki, 0, -1]
        if (p > -1).any():
            ap50[cid - 1] = p[p > -1].mean()
    return {"mAP50_95": float(E.stats[0]), "mAP50": float(E.stats[1])}, ap50


# --------------------------------------------------------------------------- image level

def per_image_table(test, gt_by_uid, dets, names, thr):
    rows = []
    for r in test.itertuples():
        d = dets.get(r.uid, [])
        top = max(d, key=lambda x: x[5]) if d else None
        pred = names[top[4]] if top else "empty"
        score = top[5] if top else 0.0
        pred_count = sum(1 for x in d if top and x[4] == top[4] and x[5] >= thr)
        rows.append({"uid": r.uid, "split": r.split, "yolov4_seen": int(getattr(r, "yolov4_seen", 0) or 0),
                     "location": r.location, "datetime": r.datetime, "season": r.season,
                     "true_label": r.label, "pred_label": pred, "score": score,
                     "true_count": len(gt_by_uid.get(r.uid, [])), "pred_count": pred_count,
                     "night": int(r.night), "ir": r.ir, "vegetation": r.vegetation,
                     "rain": int(r.poor_visibility), "poor_visibility": int(r.poor_visibility)})
    return pd.DataFrame(rows)


def image_metrics(pi, thr, names):
    called = pi.score >= thr
    pred = np.where(called, pi.pred_label, "empty")
    true = pi.true_label.to_numpy()
    rows = []
    for c in names:
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        p, r, f = prf(np.array([tp], float), np.array([fp], float), np.array([fn], float))
        rows.append({"species": c, "img_tp": tp, "img_fp": fp, "img_fn": fn,
                     "img_precision": p[0], "img_recall": r[0], "img_f1": f[0], "n_test_images": int((true == c).sum())})
    df = pd.DataFrame(rows)
    present = df[df.n_test_images > 0]
    animal = true != "empty"
    out = {
        "img_macro_precision": float(present.img_precision.mean(skipna=True)),
        "img_macro_recall": float(present.img_recall.mean(skipna=True)),
        "img_macro_f1": float(present.img_f1.fillna(0).mean()),
        "species_accuracy_on_animal_images": float((pred[animal] == true[animal]).mean()) if animal.any() else np.nan,
        "empty_filter_animal_recall": float(called[animal].mean()) if animal.any() else np.nan,
        "empty_filter_animal_precision": float(animal[called].mean()) if called.any() else np.nan,
        "empty_false_positive_rate": float(called[~animal].mean()) if (~animal).any() else np.nan,
    }
    return out, df, pred


def condition_table(pi, thr):
    pi = pi.copy()
    pi["correct"] = (pi.score >= thr) & (pi.pred_label == pi.true_label)
    pi["called"] = pi.score >= thr
    terc = pd.qcut(pi.vegetation.rank(method="first"), 3, labels=["open", "medium", "dense"])
    strata = {"time": np.where(pi.night == 1, "night", "day"), "season": pi.season,
              "vegetation": terc.astype(str), "visibility": np.where(pi.poor_visibility == 1, "poor", "good")}
    rows = []
    for var, lev in strata.items():
        pi["_lev"] = lev
        for level, g in pi.groupby("_lev"):
            an, em = g[g.true_label != "empty"], g[g.true_label == "empty"]
            rows.append({"factor": var, "level": level, "n_animal": len(an), "n_empty": len(em),
                         "recall": an.correct.mean() if len(an) else np.nan,
                         "empty_fp_rate": em.called.mean() if len(em) else np.nan})
    return pd.DataFrame(rows)


def ecology_checks(pi, thr):
    called = pi.score >= thr
    pi = pi.assign(pred_eff=np.where(called, pi.pred_label, "empty"))
    true_sets = pi[pi.true_label != "empty"].groupby("location").true_label.agg(set)
    pred_sets = pi[pi.pred_eff != "empty"].groupby("location").pred_eff.agg(set)
    locs = sorted(set(pi.location))
    t = [true_sets.get(l, set()) for l in locs]
    p = [pred_sets.get(l, set()) for l in locs]
    rich_err = [len(b) - len(a) for a, b in zip(t, p)]
    jac = [len(a & b) / len(a | b) if a | b else 1.0 for a, b in zip(t, p)]
    species = sorted(set().union(*t)) if t else []
    occ = []
    for s in species:
        occ.append({"species": s, "true_naive_occ": np.mean([s in a for a in t]),
                    "pred_naive_occ": np.mean([s in b for b in p])})
    occ = pd.DataFrame(occ)
    ok = (pi.pred_eff == pi.true_label) & (pi.true_label != "empty")
    return {
        "richness_mean_error_per_site": float(np.mean(rich_err)) if locs else np.nan,
        "richness_mean_abs_error_per_site": float(np.mean(np.abs(rich_err))) if locs else np.nan,
        "species_set_jaccard_per_site": float(np.mean(jac)) if locs else np.nan,
        "naive_occupancy_mean_abs_error": float((occ.pred_naive_occ - occ.true_naive_occ).abs().mean()) if len(occ) else np.nan,
        "count_mae_correct_images": float((pi.pred_count[ok] - pi.true_count[ok]).abs().mean()) if ok.any() else np.nan,
        "n_test_sites": len(locs),
    }, occ


# --------------------------------------------------------------------------- driver

def evaluate_model(name, paths, ec, test, gt_by_uid, names, n_train):
    pr = P.load(paths, name)
    dets = {u: [list(x[:4]) + [int(x[4]), float(x[5])] for x in v] for u, v in pr["detections"].items()}
    k = len(names)
    thr = ec["threshold"]
    out_dir = paths.eval / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Box-level metrics (mAP, box P/R/F1) use the box-annotated test split only.
    boxed_uids = set(test[test.split == "test"].uid)
    dets_box = {u: v for u, v in dets.items() if u in boxed_uids}
    m_coco, ap50 = coco_map(paths, dets_box, k)
    tp, fp, fn = box_counts(gt_by_uid, dets_box, thr, ec["iou"], k)
    bp, br, bf = prf(tp, fp, fn)
    # Image-level and ecological metrics use the "clean" test images: boxed test, empty and
    # independent image-level test images that the YOLOv4 reference never trained on.
    pi_all = per_image_table(test, gt_by_uid, dets, names, thr)
    pi = pi_all[pi_all.yolov4_seen == 0].reset_index(drop=True)
    m_img, sp_img, pred_eff = image_metrics(pi, thr, names)
    m_boxed, _, _ = image_metrics(pi_all[pi_all.split == "test"], thr, names)
    m_eco, occ = ecology_checks(pi, thr)

    sp = pd.DataFrame({"species": names, "n_train_images": [n_train.get(c, 0) for c in names],
                       "ap50": ap50, "box_precision": bp, "box_recall": br, "box_f1": bf,
                       "box_tp": tp, "box_fp": fp, "box_fn": fn}).merge(sp_img, on="species")
    has_gt = (tp + fn) > 0
    metrics = {"model": name, "threshold": thr, **m_coco,
               "box_macro_precision": float(np.nanmean(bp[has_gt])) if has_gt.any() else np.nan,
               "box_macro_recall": float(np.nanmean(br[has_gt])) if has_gt.any() else np.nan,
               "box_macro_f1": float(np.nan_to_num(bf[has_gt]).mean()) if has_gt.any() else np.nan,
               **m_img, **{f"boxedtest_{k_}": v for k_, v in m_boxed.items()}, **m_eco,
               "n_clean_test_images": len(pi), "n_boxed_test_images": int((pi_all.split == "test").sum()),
               "n_clean_imagelevel_images": int((pi.split == "test_il").sum()),
               "timing": pr.get("timing", {})}

    sweep = []
    for t in ec["sweep"]:
        mi, _, _ = image_metrics(pi, t, names)
        sweep.append({"threshold": t, **mi})
    labels = names + ["empty"]
    conf = pd.crosstab(pd.Categorical(pi.true_label, labels), pd.Categorical(pred_eff, labels),
                       rownames=["true"], colnames=["pred"], dropna=False)

    write_json(metrics, out_dir / "metrics.json")
    sp.to_csv(out_dir / "per_species.csv", index=False)
    pi_all.to_csv(out_dir / "per_image.csv", index=False)
    condition_table(pi, thr).to_csv(out_dir / "conditions.csv", index=False)
    pd.DataFrame(sweep).to_csv(out_dir / "threshold_sweep.csv", index=False)
    conf.to_csv(out_dir / "confusion.csv")
    occ.to_csv(out_dir / "naive_occupancy.csv", index=False)
    log.info("%s: mAP50 %.3f  mAP50-95 %.3f  image macro-F1 %.3f  empty-FP %.3f", name,
             metrics["mAP50"], metrics["mAP50_95"], metrics["img_macro_f1"], metrics["empty_false_positive_rate"])
    return metrics


def figures(paths, models):
    col = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}
    sp = {m: pd.read_csv(paths.eval / m / "per_species.csv") for m in models}
    cond = {m: pd.read_csv(paths.eval / m / "conditions.csv") for m in models}
    sweep = {m: pd.read_csv(paths.eval / m / "threshold_sweep.csv") for m in models}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in models:
        d = sp[m][sp[m].n_test_images > 0]
        ax.scatter(np.maximum(d.n_train_images, 1), d.img_f1.fillna(0), color=col[m], label=m, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("training images per species (log)")
    ax.set_ylabel("image-level F1 (test)")
    ax.set_title("Rare species are where models differ")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.eval / "f1_vs_training_size.png", dpi=150)
    plt.close(fig)

    c0 = cond[models[0]]
    keys = list(zip(c0.factor, c0.level))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        c = cond[m].set_index(["factor", "level"]).reindex(keys)
        ax.bar(np.arange(len(keys)) + i * w, c.recall, w, color=col[m], label=m)
    ax.set_xticks(np.arange(len(keys)) + 0.4 - w / 2)
    ax.set_xticklabels([f"{f}\n{l}" for f, l in keys], fontsize=8)
    ax.set_ylabel("image-level recall (correct species)")
    ax.set_ylim(0, 1)
    ax.set_title("Recall under different environmental conditions (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.eval / "recall_by_condition.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in models:
        s = sweep[m]
        axes[0].plot(s.img_macro_recall, s.img_macro_precision, "o-", color=col[m], label=m)
        axes[1].plot(s.threshold, s.empty_false_positive_rate, "o-", color=col[m], label=m)
    axes[0].set_xlabel("macro recall")
    axes[0].set_ylabel("macro precision")
    axes[0].set_title("Precision-recall across thresholds")
    axes[1].set_xlabel("confidence threshold")
    axes[1].set_ylabel("share of empty images given a species")
    axes[1].set_title("False positives on empty frames")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.eval / "threshold_curves.png", dpi=150)
    plt.close(fig)


def main():
    args, cfg, paths = cli(__doc__, lambda ap: ap.add_argument("--models", nargs="*"))
    ec = cfg["evaluate"]
    names = load_classes(paths)
    n_train = read_json(paths.classes)["n_train_images"]
    test = P.test_images(paths)
    bx = pd.read_csv(paths.boxes)
    bx = bx[bx.split == "test"]
    gt_by_uid = defaultdict(list)
    for b in bx.itertuples():
        gt_by_uid[b.uid].append([b.x, b.y, b.w, b.h, int(b.cls)])

    models = [m for m in (args.models or ec["models"]) if (paths.preds / f"{m}.json").exists()]
    missing = set(args.models or ec["models"]) - set(models)
    if missing:
        log.warning("no predictions for: %s (skipped)", ", ".join(sorted(missing)))
    if not models:
        raise SystemExit("no predictions found in " + str(paths.preds))
    rows = []
    for m in models:
        met = evaluate_model(m, paths, ec, test, gt_by_uid, names, n_train)
        t = met.pop("timing")
        rows.append({**met, "ms_per_image": t.get("ms_per_image"), "cpu_ms_per_image": t.get("cpu_ms_per_image"),
                     "kwh_per_1000_images": (t["kwh_per_image"] * 1000) if t.get("kwh_per_image") else None})
    comp = pd.DataFrame(rows)
    comp.to_csv(paths.eval / "comparison.csv", index=False)
    figures(paths, models)
    log.info("comparison:\n%s", comp[["model", "mAP50", "mAP50_95", "img_macro_precision", "img_macro_recall",
                                      "img_macro_f1", "empty_false_positive_rate", "ms_per_image"]].round(3).to_string())


if __name__ == "__main__":
    main()
