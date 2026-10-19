"""Run a trained Ultralytics model (YOLO11 / RT-DETR) on the test split.

    python -m nkpipe.predict_ultralytics --model YOLO11
    python -m nkpipe.predict_ultralytics --model RT-DETR
"""

import time

from . import cli, load_classes, log, read_json
from .energy import EnergyMeter
from . import preds as P


def run(model, paths_list, imgsz, pc, device):
    dets, speed = {}, []
    for i in range(0, len(paths_list), 256):
        chunk = paths_list[i:i + 256]
        for uid, r in zip([u for u, _ in chunk], model.predict(
                [p for _, p in chunk], imgsz=imgsz, conf=pc["conf"], iou=pc["iou"], device=device,
                batch=pc["batch"], stream=True, verbose=False)):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy()
            dets[uid] = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1), int(c), float(s)]
                         for (x1, y1, x2, y2), c, s in zip(xyxy, b.cls.cpu().numpy(), b.conf.cpu().numpy())]
            speed.append(sum(r.speed.values()))
    return dets, speed


def main():
    args, cfg, paths = cli(__doc__, lambda ap: ap.add_argument("--model", required=True,
                                                               choices=["YOLO11", "RT-DETR"]))
    from ultralytics import RTDETR, YOLO

    info = read_json(paths.models / f"{args.model}.json")
    cls = RTDETR if args.model == "RT-DETR" else YOLO
    model = cls(info["weights"])
    pc = cfg["predict"]
    test = P.test_images(paths)
    items = list(zip(test.uid, test.path))
    device = cfg["train"]["device"]
    gpu = None if device == "cpu" else str(device)

    with EnergyMeter(device_index=gpu) as em:
        dets, speed = run(model, items, info["imgsz"], pc, device)
    timing = P.timing_from_energy(em.summary(), len(items))
    timing["ultralytics_ms_per_image"] = sum(speed) / max(len(speed), 1)

    n_cpu = min(pc.get("cpu_benchmark_images") or 0, len(items))
    if n_cpu:
        cpu_model = cls(info["weights"])
        t0 = time.time()
        run(cpu_model, items[:n_cpu], info["imgsz"], {**pc, "batch": 1}, "cpu")
        timing["cpu_ms_per_image"] = 1000 * (time.time() - t0) / n_cpu

    P.save(paths, args.model, load_classes(paths), dets, timing)
    log.info("%s: %d test images, %.1f ms/image (GPU)", args.model, len(items), timing["ms_per_image"])


if __name__ == "__main__":
    main()
