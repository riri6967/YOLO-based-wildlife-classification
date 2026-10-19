# Running the full study on Ubuntu 24.04 (ROS 2 Jazzy machine)

This runs everything, from downloading the data to the ecological simulation:

```
download ─► prepare ─► train YOLO11 ─┐
                      train RT-DETR ─┤
                      train classifier ─► MegaDetector ─┐
                                                        ├─► predict (4 models) ─► evaluate ─► simulate
          YOLOv4 reference (Zenodo, Appel et al. 2025) ─┘
```

## 1. What you need

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, 8 GB (lower `imgsz`/`batch`) | 16–24 GB (RTX 3090/4090/A5000) |
| Disk | 60 GB free | 100 GB (SSD) |
| RAM | 16 GB | 32 GB |
| Driver | NVIDIA driver ≥ 535 (`nvidia-smi` works) | |

## 2. Install (once)

```bash
git clone -b claude/nifty-meitner-hou0x1 https://github.com/riri6967/YOLO-based-wildlife-classification.git
cd YOLO-based-wildlife-classification
bash setup/install_ubuntu.sh
```

The script creates `.venv`, used for training, evaluation and simulation, and
`.venv-md`, used only for MegaDetector. **ROS 2 note:** if your `.bashrc`
sources `/opt/ros/jazzy/setup.bash`, the ROS Python packages end up on
`PYTHONPATH`. The scripts unset it so ROS cannot shadow numpy or OpenCV.

## 3. Trial run first (about 1–2 h)

```bash
bash run_all.sh --quick
```

This uses about 3,000 images and a few epochs (`configs/quick.yaml`) and
writes to `work_quick/` and `results_quick/`. It checks that the download,
GPU, MegaDetector and every stage work on your machine. The training logs
show the time per epoch, so you can estimate how long the full run will take.

## 4. Full run

```bash
bash run_all.sh                     # everything, resumable stage by stage
bash run_all.sh --from evaluate     # resume from a stage
bash run_all.sh --only train_rtdetr # run one stage
```

Long training runs are safest inside `tmux` (or `nohup`), so a closed
terminal doesn't kill them:
```bash
sudo apt install tmux && tmux new -s wildlife
bash run_all.sh
# detach: Ctrl-b d      reattach: tmux attach -t wildlife
```
If YOLO11 or RT-DETR training is interrupted, resume it with
`.venv/bin/python -m nkpipe.train_ultralytics --model yolo11 --resume`.

**Settings in `configs/pipeline.yaml` you may want to change:**
- `train.yolo11.imgsz` and `train.rtdetr.imgsz`: 960 by default, because
  animals are often small in these frames. If you run out of GPU memory, use
  640 or lower `batch`.
- `train.*.epochs`: training stops early at `patience` once validation stops
  improving.
- `simulate.energy_kgco2_per_kwh`: the grid carbon intensity of the country
  where **you** train and run the models. It converts measured kWh into CO2.
- `simulate.priority_species_keywords`: which species count as priority for
  alerts.

## 5. What each stage does

| Stage | Output | Notes |
|---|---|---|
| `download` | `work/raw/` | Finds the metadata through LILA's index. If that fails, paste the metadata link from the [LILA page](https://lila.science/datasets/nkhotakota-camera-traps/) into `download.metadata_urls`. Downloads all ~33.8k box-annotated images plus 10k empty images (~30 GB), and the YOLOv4 reference from Zenodo |
| `prepare` | `work/data/` | **Camera-site-disjoint split** (no site appears in both train and test). Per image it computes infrared/night, season, vegetation density per site, and a poor-visibility proxy (rain/fog/blur). Builds the YOLO dataset, the COCO test ground truth and the classifier crops |
| `train_yolo11`, `train_rtdetr` | `work/runs/` | Ultralytics; GPU energy measured through NVML → `work/energy/` |
| `train_classifier` | `work/models/classifier.pt` | timm ConvNeXt-T on crops; the sampler is balanced toward rare species; random greyscale augmentation mimics IR |
| `megadetector` | `work/preds/megadetector_raw.json` | MDv5a, run in `.venv-md` |
| `predict` | `work/preds/<model>.json` | One common format for all four models, with timing and energy per image |
| `evaluate` | `results/eval/` | See below |
| `simulate` | `results/simulation/` | Ecology fitted from the data, error profiles fitted from the predictions, then the scenario experiment and Sobol sensitivity analysis |

## 6. Results to use in the paper

- `results/eval/comparison.csv`: per model, mAP50 and mAP50-95; box- and
  image-level macro precision/recall/F1 (directly comparable with Appel et
  al.'s P = 0.64, R = 0.76, F1 = 0.63); empty-image filtering; false-positive
  rate on empty frames; count error; per-site richness error and Jaccard;
  naive occupancy error on test sites; ms/image; kWh per 1,000 images.
- `results/eval/<model>/per_species.csv`: AP50 and F1 against the number of
  training images (shows the rare-species problem).
- `results/eval/<model>/conditions.csv`: recall and false positives by
  day/night, wet/dry season, vegetation density and visibility. This is the
  **environment → detection** result.
- `results/eval/*.png`: F1 vs training size, recall by condition, and
  threshold curves.
- `results/simulation/`:
  - `species_ecology_from_data.csv`: occupancy (MacKenzie), event rates and
    night activity estimated from all 321k labelled images.
  - `calibrated_config.json`: the fitted detector profiles.
  - `summary.csv` and figures: occupancy bias, missed priority events, alert
    latency and CO2 per model, threshold, season and monitoring regime.
  - `sobol_indices.csv` and `.png`: which factors matter most.

## 7. Caveats to state in the paper

1. **YOLOv4 reference leakage:** NWR_YOLO_v1 was trained on Nkhotakota
   images, and its training split is not public, so some of our test images
   may have been in its training set. Report it as an optimistic reference,
   not a clean comparison. It also uses a different class list; unmatched
   classes are listed in `work/preds/yolov4_class_mapping.json`.
2. **The environmental covariates are automatic proxies.** Night/IR is
   reliable. Vegetation (green-pixel share per site) and poor visibility (low
   contrast or blur) are approximate; hand-label ~300 test images to validate
   them.
3. **Energy is GPU board power only** (from NVML). CPU, RAM and cooling are
   not included, so it is a lower bound; the simulation applies a PUE factor
   to training.
4. Monitoring-regime and vehicle parameters in the simulation are still
   assumptions (`configs/default.json`). Replace them with values from the
   reserve's operations if you can.

## 8. Checking the code without a GPU

```bash
.venv/bin/python -m pytest -q tests           # unit tests
bash tests/smoke_pipeline.sh                  # every stage on synthetic data, CPU, ~15 min
```
