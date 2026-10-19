"""Energy use and CO2-equivalent emissions of running a monitoring programme
with a given detector and field regime.

Components:
  * inference  -- images x ms/image x device power x grid carbon intensity
  * training   -- GPU hours x power x PUE, amortised over several surveys
  * field      -- vehicle km for SD-card servicing rounds and alert responses
  * telemetry  -- energy to transmit images from cellular cameras
"""


def survey_footprint(cfg, model_name, n_images, alerts, regime_name):
    fp = cfg["footprint"]
    prof = cfg["models"][model_name]
    reg = cfg["monitoring_regimes"][regime_name]
    dep = fp["deployment"]

    if dep == "gpu" and prof.get("infer_kwh_per_image") is not None:  # measured on real hardware
        infer_kwh = n_images * prof["infer_kwh_per_image"]
    else:
        infer_kwh = n_images * prof["infer_ms"][dep] / 1000 * fp["device_watts"][dep] / 3.6e6
    infer_co2 = infer_kwh * fp["grid_kgco2_per_kwh"][dep]

    if prof.get("train_kwh") is not None:
        train_kwh = prof["train_kwh"] * fp["train_pue"] / fp["amortise_training_over_surveys"]
    else:
        train_kwh = (prof["train_gpu_hours"] * fp["train_gpu_watts"] * fp["train_pue"] / 1000
                     / fp["amortise_training_over_surveys"])
    train_co2 = train_kwh * fp["train_kgco2_per_kwh"]

    km = alerts["n_checks"] * fp["sd_check_route_km"] + alerts["response_trips"] * fp["response_trip_km"]
    field_co2 = km * fp["vehicle_kgco2_per_km"]

    tele_kwh = n_images * fp["cellular_wh_per_image"] / 1000 if reg["check_interval_days"] == 0 else 0.0
    tele_co2 = tele_kwh * fp["cellular_kgco2_per_kwh"]

    return {
        "compute_kwh": infer_kwh + train_kwh,
        "co2_inference_kg": infer_co2,
        "co2_training_kg": train_co2,
        "co2_field_kg": field_co2,
        "co2_telemetry_kg": tele_co2,
        "co2_total_kg": infer_co2 + train_co2 + field_co2 + tele_co2,
        "vehicle_km": km,
        "processing_hours": n_images * prof["infer_ms"][dep] / 3.6e6,
    }
