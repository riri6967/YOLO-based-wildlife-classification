"""Operational monitoring outcomes for priority species (e.g. elephants for
human-wildlife conflict, large carnivores): missed events, false alarms and
how quickly a true event turns into a field response."""

import numpy as np


def alert_outcomes(cfg, world, obs, regime_name, rng):
    reg = cfg["monitoring_regimes"][regime_name]
    pri = np.array([s["priority"] for s in cfg["species"]])
    S, D, _ = obs.correct.shape

    true_ev = world.n_passages[..., pri] > 0           # (S, D, Kp) site-days with an animal
    hit = obs.correct[..., pri] > 0
    false_alarm = (obs.false_pos[..., pri] > 0) & ~hit
    # A ranger screens each alert image; screened-out false alarms cost no trip.
    dispatched_fa = false_alarm & (rng.random(false_alarm.shape) >= cfg["alert_screening_catch"])
    # Only some true events need a field response (e.g. animals near the boundary).
    dispatched_hit = hit & (rng.random(hit.shape) < cfg["true_alert_dispatch_frac"])

    # Time of the first correctly detected passage within the day.
    n_hit = np.maximum(obs.correct[..., pri], 1)
    t_event = np.arange(D)[None, :, None] + rng.beta(1, n_hit)

    C = reg["check_interval_days"]
    if C > 0:
        offset = rng.uniform(0, C, size=(S, 1, 1))     # staggered servicing rounds
        wait_days = np.ceil((t_event - offset) / C) * C + offset - t_event
    else:
        wait_days = np.zeros_like(t_event)
    latency_h = (wait_days * 24 + reg["transmit_hours"]
                 + reg["processing_hours"] + reg["response_hours"])

    lat = latency_h[hit]
    n_true = int(true_ev.sum())
    timely = int((hit & (latency_h <= cfg["alert_deadline_hours"])).sum())
    return {
        "priority_event_days": n_true,
        "missed_event_frac": 1 - hit.sum() / n_true if n_true else np.nan,
        "false_alarms": int(false_alarm.sum()),
        "false_alarm_share": false_alarm.sum() / max(hit.sum() + false_alarm.sum(), 1),
        "median_latency_h": float(np.median(lat)) if lat.size else np.nan,
        "timely_response_frac": timely / n_true if n_true else np.nan,
        "response_trips": int(dispatched_hit.sum() + dispatched_fa.sum()),
        "n_checks": int(np.ceil(D / (C if C > 0 else reg["service_interval_days"]))),
    }
