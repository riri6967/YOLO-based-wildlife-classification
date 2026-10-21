import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yaml")

from nkpipe import _merge  # noqa: E402
from nkpipe.download import merge_coco, relative_name  # noqa: E402
from nkpipe.evaluate import box_counts, ecology_checks, image_metrics, iou_xywh  # noqa: E402
from nkpipe.prepare import norm_name, split_locations, uid_of  # noqa: E402


def test_config_merge_overrides_nested_keys():
    base = {"a": 1, "train": {"yolo11": {"epochs": 100, "imgsz": 960}}}
    out = _merge(base, {"train": {"yolo11": {"epochs": 5}}})
    assert out["train"]["yolo11"] == {"epochs": 5, "imgsz": 960} and out["a"] == 1


def test_relative_name_and_uid():
    dl = {"image_base_url": "https://x/public-datasets-lila/nkhotakota-camera-traps/"}
    assert relative_name("nkhotakota-camera-traps/B10/a.JPG", dl) == "B10/a.JPG"
    assert relative_name("B10/a.JPG", dl) == "B10/a.JPG"
    assert uid_of("B10/B10-A/x_(1).JPG") == "B10__B10-A__x_(1)"
    assert norm_name("Sable Antelope") == "sable_antelope"


def test_merge_coco_dedupes_annotations():
    a = {"images": [{"id": 1, "file_name": "f"}], "categories": [{"id": 2, "name": "x"}],
         "annotations": [{"image_id": 1, "category_id": 2}]}
    b = {"images": [{"id": 1, "width": 10}], "annotations": [{"image_id": 1, "category_id": 2},
                                                             {"image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1]}]}
    m = merge_coco([a, b])
    assert len(m["images"]) == 1 and m["images"][0]["width"] == 10
    assert len(m["annotations"]) == 2


def test_split_is_site_disjoint_and_covers_classes():
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.integers(0, 20, (30, 5)), index=[f"L{i}" for i in range(30)],
                          columns=list("abcde"))
    assign = split_locations(counts, {"train": 0.7, "val": 0.15, "test": 0.15}, 100, 1)
    assert set(assign) == set(counts.index)
    for s in ("val", "test"):
        locs = [l for l, v in assign.items() if v == s]
        assert (counts.loc[locs].sum() > 0).all()
    frac = counts.loc[[l for l, v in assign.items() if v == "test"]].values.sum() / counts.values.sum()
    assert 0.08 < frac < 0.25


def test_iou_and_box_matching():
    assert np.isclose(iou_xywh([0, 0, 10, 10], [[0, 0, 10, 10]])[0, 0], 1.0)
    assert np.isclose(iou_xywh([0, 0, 10, 10], [[5, 0, 10, 10]])[0, 0], 50 / 150)
    gt = {"a": [[0, 0, 10, 10, 0]], "b": [[0, 0, 10, 10, 1]]}
    dets = {"a": [[0, 0, 10, 10, 0, 0.9], [50, 50, 5, 5, 0, 0.8]],  # TP + FP
            "b": [[0, 0, 10, 10, 0, 0.9]]}                             # wrong class: FP(0) + FN(1)
    tp, fp, fn = box_counts(gt, dets, 0.5, 0.5, 2)
    assert tp.tolist() == [1, 0] and fp.tolist() == [2, 0] and fn.tolist() == [0, 1]


def test_image_metrics_and_ecology():
    pi = pd.DataFrame({
        "uid": list("abcd"), "location": ["L1", "L1", "L2", "L2"],
        "true_label": ["x", "empty", "y", "x"], "pred_label": ["x", "x", "x", "x"],
        "score": [0.9, 0.6, 0.7, 0.3], "true_count": [1, 0, 1, 1], "pred_count": [1, 1, 1, 0]})
    m, df, _ = image_metrics(pi, 0.5, ["x", "y"])
    assert m["empty_false_positive_rate"] == 1.0
    assert df.set_index("species").loc["x", "img_tp"] == 1
    eco, occ = ecology_checks(pi, 0.5)
    assert eco["n_test_sites"] == 2
    assert occ.set_index("species").loc["y", "pred_naive_occ"] == 0.0


def test_yolov4_class_matching():
    from nkpipe.predict_yolov4 import match_class
    ours = ["african_savanna_elephant", "greater_kudu", "cape_buffalo", "southern_ground_hornbill",
            "helmeted_guineafowl", "side_striped_jackal", "domestic_dog", "small_antelope", "yellow_baboon"]
    assert match_class("elephant", ours, {}) == "african_savanna_elephant"
    assert match_class("kudu", ours, {}) == "greater_kudu"
    assert match_class("ground_hornbill", ours, {}) == "southern_ground_hornbill"
    assert match_class("guinea_fowl", ours, {}) == "helmeted_guineafowl"
    assert match_class("side-striped_jackal", ours, {}) == "side_striped_jackal"
    assert match_class("dog", ours, {}) == "domestic_dog"
    assert match_class("duiker", ours, {}) is None
    assert match_class("duiker", ours, {"duiker": "small_antelope"}) == "small_antelope"
