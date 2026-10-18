"""Figures for the model comparison."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _colors(models):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def occupancy_bias(spec, out, threshold=0.5):
    d = spec[spec.threshold == threshold]
    models = list(d.model.unique())
    col = _colors(models)
    order = d.groupby("species").n_train.first().sort_values().index
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for i, season in enumerate(sorted(d.season.unique())):
        for j, est in enumerate(["mackenzie", "royle_link"]):
            ax = axes[i, j]
            g = d[d.season == season].groupby(["species", "model"])[f"bias_{est}"].mean().unstack()
            g = g.loc[order, models]
            w = 0.8 / len(models)
            for k, m in enumerate(models):
                ax.bar([x + k * w for x in range(len(order))], g[m], w, label=m, color=col[m])
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks([x + 0.4 - w / 2 for x in range(len(order))])
            ax.set_xticklabels(order, rotation=45, ha="right")
            ax.set_title(f"{season} season - {est.replace('_', '-')} estimator")
            if j == 0:
                ax.set_ylabel("occupancy bias (est - true)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Occupancy bias by species (rarest left), threshold {threshold}")
    fig.tight_layout()
    fig.savefig(Path(out) / "occupancy_bias_by_species.png", dpi=150)
    plt.close(fig)


def environmental_recall(scen, out, threshold=0.5):
    d = scen[(scen.threshold == threshold)].drop_duplicates(["season", "rep", "model", "threshold"])
    cols = ["recall_day", "recall_night", "recall_open_veg", "recall_dense_veg", "recall_dry_day", "recall_rain"]
    labels = ["day", "night (IR)", "open veg", "dense veg", "no rain", "rain"]
    models = list(d.model.unique())
    col = _colors(models)
    g = d.groupby("model")[cols].mean().loc[models]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        ax.bar([x + k * w for x in range(len(cols))], g.loc[m], w, label=m, color=col[m])
    ax.set_xticks([x + 0.4 - w / 2 for x in range(len(cols))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("recall of animal passages")
    ax.set_ylim(0, 1)
    ax.set_title("Detector recall under different environmental conditions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out) / "recall_by_environment.png", dpi=150)
    plt.close(fig)


def threshold_tradeoff(scen, out, regime="cellular_realtime"):
    d = scen[scen.regime == regime]
    models = list(d.model.unique())
    col = _colors(models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, season in zip(axes, sorted(d.season.unique())):
        g = d[d.season == season].groupby(["model", "threshold"])[["missed_event_frac", "false_alarms"]].mean()
        for m in models:
            gm = g.loc[m]
            ax.plot(gm.false_alarms, gm.missed_event_frac, "o-", color=col[m], label=m)
            for t, r in gm.iterrows():
                ax.annotate(f"{t:g}", (r.false_alarms, r.missed_event_frac), fontsize=7,
                            textcoords="offset points", xytext=(3, 3))
        ax.set_xlabel("false alarms for priority species (per survey)")
        ax.set_ylabel("fraction of priority-species events missed")
        ax.set_title(f"{season} season (labels = confidence threshold)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out) / "threshold_tradeoff.png", dpi=150)
    plt.close(fig)


def carbon_vs_accuracy(scen, out, threshold=0.5):
    d = scen[scen.threshold == threshold]
    g = d.groupby(["model", "regime"])[["co2_total_kg", "mae_occ_royle_link",
                                        "timely_response_frac"]].mean().reset_index()
    models = list(g.model.unique())
    col = _colors(models)
    marker = {r: mk for r, mk in zip(sorted(g.regime.unique()), "osD^v")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for _, r in g.iterrows():
        for ax, y in zip(axes, ["mae_occ_royle_link", "timely_response_frac"]):
            ax.scatter(r.co2_total_kg, r[y], color=col[r.model], marker=marker[r.regime], s=60)
    axes[0].set_ylabel("mean |occupancy bias| (Royle-Link)")
    axes[1].set_ylabel("priority events answered within deadline")
    for ax in axes:
        ax.set_xlabel("survey CO2e (kg): compute + training share + vehicles + telemetry")
    handles = [plt.Line2D([], [], color=col[m], marker="o", ls="", label=m) for m in models]
    handles += [plt.Line2D([], [], color="grey", marker=marker[r], ls="", label=r) for r in marker]
    axes[1].legend(handles=handles, fontsize=7)
    fig.suptitle("Environmental cost vs ecological performance")
    fig.tight_layout()
    fig.savefig(Path(out) / "carbon_vs_performance.png", dpi=150)
    plt.close(fig)


def sobol_indices(si, out):
    outputs = list(si.output.unique())
    fig, axes = plt.subplots(1, len(outputs), figsize=(3.2 * len(outputs), 4.5), sharey=True)
    for ax, o in zip(axes, outputs):
        d = si[si.output == o].set_index("parameter")
        ax.barh(d.index, d.ST.clip(lower=0), xerr=d.ST_conf, color="#4C72B0", alpha=0.85, label="total (ST)")
        ax.barh(d.index, d.S1.clip(lower=0), color="#DD8452", alpha=0.9, height=0.4, label="first-order (S1)")
        ax.set_title(o, fontsize=9)
        ax.set_xlim(0, 1)
    axes[0].legend(fontsize=7)
    fig.suptitle("Sobol sensitivity of monitoring outcomes")
    fig.tight_layout()
    fig.savefig(Path(out) / "sobol_indices.png", dpi=150)
    plt.close(fig)
