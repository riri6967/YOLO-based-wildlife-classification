"""Train YOLO11 or RT-DETR with Ultralytics on the prepared dataset,
measuring GPU time and energy.

    python -m nkpipe.train_ultralytics --model yolo11
    python -m nkpipe.train_ultralytics --model rtdetr
"""

from . import cli, log, write_json
from .energy import EnergyMeter

MODEL_NAMES = {"yolo11": "YOLO11", "rtdetr": "RT-DETR"}


def main():
    args, cfg, paths = cli(__doc__, lambda ap: (
        ap.add_argument("--model", required=True, choices=list(MODEL_NAMES)),
        ap.add_argument("--resume", action="store_true")))
    from ultralytics import RTDETR, YOLO

    tc = cfg["train"][args.model]
    device = cfg["train"]["device"]
    name = MODEL_NAMES[args.model]
    run_dir = paths.runs / args.model
    last = run_dir / "weights" / "last.pt"

    if args.resume and last.exists():
        model = (RTDETR if args.model == "rtdetr" else YOLO)(str(last))
        train_kw = {"resume": True}
    else:
        model = (RTDETR if args.model == "rtdetr" else YOLO)(tc["weights"])
        train_kw = dict(data=str(paths.yolo / "data.yaml"), epochs=tc["epochs"], imgsz=tc["imgsz"],
                        batch=tc["batch"], patience=tc["patience"], workers=tc["workers"],
                        device=device, seed=cfg["seed"], project=str(paths.runs), name=args.model,
                        exist_ok=True, plots=True, cos_lr=True, close_mosaic=10)
        for k, v in tc.get("extra", {}).items():
            train_kw[k] = v

    gpu = None if device == "cpu" else str(device).split(",")[0]
    with EnergyMeter(device_index=gpu) as em:
        model.train(**train_kw)
    best = run_dir / "weights" / "best.pt"
    em.save(paths.energy / f"{name}_train.json", model=name, weights=str(best),
            epochs=tc["epochs"], imgsz=tc["imgsz"])
    write_json({"model": name, "weights": str(best), "imgsz": tc["imgsz"]}, paths.models / f"{name}.json")
    log.info("%s trained: %s", name, best)


if __name__ == "__main__":
    main()
