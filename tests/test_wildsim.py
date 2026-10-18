import numpy as np

from wildsim import load_config
from wildsim import occupancy as occ
from wildsim.calibrate import fit_profile, synthetic_predictions
from wildsim.detection import observe
from wildsim.footprint import survey_footprint
from wildsim.world import simulate_world


def _histories(psi, p, S=400, J=8, seed=0, p10=0.0):
    rng = np.random.default_rng(seed)
    z = rng.random(S) < psi
    pr = np.where(z[:, None], p, p10)
    return (rng.random((S, J)) < pr).astype(int)


def test_mackenzie_recovers_psi():
    y = _histories(0.4, 0.3)
    assert abs(occ.mackenzie(y)["psi"] - 0.4) < 0.07


def test_royle_link_handles_false_positives():
    y = _histories(0.4, 0.5, p10=0.05, S=800)
    assert abs(occ.royle_link(y)["psi"] - 0.4) < 0.08
    # The standard model absorbs false positives as extra occupied sites.
    assert occ.mackenzie(y)["psi"] > 0.5


def test_to_occasions_collapses_days():
    y = np.array([[0, 0, 1, 0, 0, 0]])
    assert occ.to_occasions(y, 3).tolist() == [[1, 0]]


def test_environment_reduces_detection():
    cfg = load_config()
    rng = np.random.default_rng(1)
    w = simulate_world(cfg, "dry", rng)
    obs_dense = observe(cfg, w, "YOLO11", 0.5, np.random.default_rng(2))
    w.veg_eff[:] = 0.0
    obs_open = observe(cfg, w, "YOLO11", 0.5, np.random.default_rng(2))
    assert obs_open.correct.sum() > obs_dense.correct.sum()


def test_common_random_numbers_share_world():
    cfg = load_config()
    w = simulate_world(cfg, "wet", np.random.default_rng(3))
    a = observe(cfg, w, "YOLO11", 0.5, np.random.default_rng(4))
    b = observe(cfg, w, "RT-DETR", 0.5, np.random.default_rng(4))
    assert a.n_images == b.n_images
    assert (a.correct <= w.n_passages).all() and (b.correct <= w.n_passages).all()


def test_footprint_arithmetic():
    cfg = load_config()
    cfg["footprint"]["deployment"] = "gpu"
    alerts = {"n_checks": 2, "response_trips": 3}
    f = survey_footprint(cfg, "YOLO11", 1_000_000, alerts, "sd_card_14d")
    fp, m = cfg["footprint"], cfg["models"]["YOLO11"]
    kwh = 1e6 * m["infer_ms"]["gpu"] / 1000 * fp["device_watts"]["gpu"] / 3.6e6
    assert np.isclose(f["co2_inference_kg"], kwh * fp["grid_kgco2_per_kwh"]["gpu"])
    km = 2 * fp["sd_check_route_km"] + 3 * fp["response_trip_km"]
    assert np.isclose(f["co2_field_kg"], km * fp["vehicle_kgco2_per_km"])
    assert f["co2_telemetry_kg"] == 0


def test_calibration_recovers_profile():
    cfg = load_config()
    prof = cfg["models"]["YOLO11"]
    fit = fit_profile(synthetic_predictions(prof, cfg["species"], n=20000, seed=5), cfg["species"])
    for k in ("recall_logit", "night_penalty", "veg_penalty", "thr_recall_slope"):
        assert abs(fit[k] - prof[k]) < 0.3, k
