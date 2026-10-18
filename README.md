# Detection uncertainty and environmental effects in automated wildlife monitoring

Simulation framework for the Nkhotakota Wildlife Reserve (Malawi) camera-trap
study. Appel et al. (2025, *Ecological Applications*, doi:10.1002/eap.70096)
trained a YOLOv4 detector on this dataset. This project goes further:

1. **Compares newer detectors** (YOLO11, RT-DETR, MegaDetector + species
   classifier) against the YOLOv4 reference.
2. **Feeds each detector's errors into ecological results**: occupancy,
   species richness, missed priority-species events and how quickly alerts
   get a response.
3. **Covers the environment in two ways**:
   - *Environment → detection*: season, rain, vegetation density and
     night-time (IR) imaging change each model's recall, and moving grass
     and rain create the empty triggers that produce false positives.
   - *Environmental footprint*: CO2e from model training, inference, cellular
     telemetry and vehicle trips (SD-card servicing rounds and alert
     responses), so ecological accuracy can be weighed against carbon cost.

> **All parameter values in `configs/default.json` are placeholder
> assumptions.** They give a working baseline. Replace the model profiles
> with values fitted from your own test-set predictions (see *Calibration*),
> and the footprint factors with sourced values, before you report results.

## How it works

```
world.py        true state per replicate: site habitat (vegetation, distance to water),
                season, rain, occupancy z ~ Bern(psi(habitat)), day/night passages,
                camera PIR triggers, empty triggers (driven by vegetation + rain)
      │  same world reused for every model (paired comparison)
detection.py    model profile -> per-image P(correct detection | env, species, threshold),
                misclassification, false positives on empty triggers
      │
occupancy.py    naive / MacKenzie (2002) / Royle-Link (2006, false-positive) occupancy
monitoring.py   missed priority events, false alarms, alert latency, timely-response rate
footprint.py    kWh and kg CO2e: training (amortised), inference, telemetry, vehicles
      │
experiment.py   grid: model x threshold x season x monitoring regime x replicates
sensitivity.py  Sobol indices over detector, environment and monitoring-design parameters
calibrate.py    fit a model profile from real per-image predictions
```

A detector profile works on the logit scale:

```
logit P(correct) = recall_logit + data_slope*log10(n_train/1000) + size_effect
                   - night_penalty*night - veg_penalty*vegetation - rain_penalty*rain
                   - thr_recall_slope*(threshold - 0.5)
P(false positive | empty trigger) = fp_per_empty * (1 + (fp_veg_mult-1)*vegetation)
                                    * exp(-thr_fp_slope*(threshold - 0.5))
```

## Running it

```bash
pip install -r requirements.txt
python -m pytest -q tests
python scripts/run_experiment.py --reps 20 --out results/
python scripts/run_sensitivity.py --n-base 128 --out results/
```

Outputs in `results/`:

| File | Content |
|---|---|
| `scenario_results.csv` | one row per season x replicate x model x threshold x regime |
| `species_results.csv` | true vs estimated occupancy per species |
| `summary.csv` | mean and SD of the key outcomes |
| `recall_by_environment.png` | recall by day/night, vegetation density and rain |
| `occupancy_bias_by_species.png` | MacKenzie vs Royle-Link bias, rare to common species |
| `threshold_tradeoff.png` | missed events vs false alarms across confidence thresholds |
| `carbon_vs_performance.png` | survey CO2e vs occupancy accuracy and response timeliness |
| `sobol_indices.png`, `sobol_indices.csv` | which parameters drive each outcome |

## Calibration: connecting the real models to the simulation

1. Train each model on Nkhotakota using a **camera-site-disjoint** split.
2. Run it on the test set and export one row per image with the columns
   `true_label, pred_label, score, night, vegetation, rain`. Night can be read
   from EXIF time or greyscale IR frames. Vegetation and rain can be scored on
   a labelled subset.
3. Fit the profile and record measured inference time and training GPU hours:

```bash
python scripts/calibrate.py --predictions preds_yolo11.csv --model YOLO11 \
    --infer-ms-gpu 12 --train-gpu-hours 30 --out configs/calibrated.json
python scripts/run_experiment.py --config configs/calibrated.json --out results_calibrated/
```

## Limitations

* The ecology is simulated. Occupancy, activity and habitat effects are
  assumptions, not estimates from Nkhotakota.
* Environmental effects are modelled as additive logit penalties. Interactions
  (e.g. night × dense vegetation) are not included.
* Occupancy models use constant psi and p, so variation in detection driven by
  the environment is deliberately left unmodelled. This reflects common
  practice but is a source of bias in itself.
* Footprint factors (grid carbon intensity, vehicle emissions, route lengths)
  depend on the site and must be sourced locally.
* Field validation is needed before the framework informs operational
  decisions.
