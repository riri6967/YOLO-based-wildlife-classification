"""Run MegaDetector (stage 1 of the two-stage pipeline) on the test split.

MegaDetector lives in its own virtualenv (.venv-md, see setup/install_ubuntu.sh)
because its dependencies differ from Ultralytics'. This wrapper writes the image
list, calls MegaDetector's batch runner in that environment, and measures GPU energy.

    python -m nkpipe.run_megadetector
"""

import shutil
import subprocess
import tempfile

from pathlib import Path

from . import ROOT, cli, log, read_json, write_json
from .energy import EnergyMeter
from . import preds as P


def main():
    args, cfg, paths = cli(__doc__, lambda ap: ap.add_argument(
        "--force", action="store_true", help="run again even if results exist"))
    mc = cfg["megadetector"]
    test = P.test_images(paths)
    done = paths.energy / "MegaDetector_infer.json"
    if done.exists() and (paths.preds / "megadetector_raw.json").exists() and not args.force:
        if read_json(done).get("n_images") == len(test):
            log.info("MegaDetector results already exist for %d test images; skipping (--force to rerun).",
                     len(test))
            return
    img_list = paths.preds / "md_input.json"
    write_json([str(p) for p in test.path], img_list)
    out = paths.preds / "megadetector_raw.json"
    py = mc["python"] if mc["python"].startswith("/") else str(ROOT / mc["python"])
    # MegaDetector downloads its weights to /tmp, which is emptied on reboot; keep a copy.
    cache = ROOT / "external" / "megadetector"
    cached = read_json(cache / "models.json").get(mc["model"]) if (cache / "models.json").exists() else None
    model_arg = str(cache / cached) if cached and (cache / cached).exists() else mc["model"]
    if model_arg != mc["model"]:
        log.info("using cached MegaDetector weights %s", model_arg)
    tmp_models = sorted((p for p in (Path(tempfile.gettempdir()) / "megadetector_models").glob("*.pt")),
                        key=lambda p: p.stat().st_mtime) if model_arg == mc["model"] else []
    cmd = [py, "-m", "megadetector.detection.run_detector_batch", model_arg, str(img_list), str(out),
           "--threshold", str(mc["threshold"]), *map(str, mc.get("extra_args", []))]
    log.info("running: %s", " ".join(cmd))
    device = cfg["train"]["device"]
    with EnergyMeter(device_index=None if device == "cpu" else str(device)) as em:
        subprocess.run(cmd, check=True)
    if model_arg == mc["model"]:
        fresh = sorted((Path(tempfile.gettempdir()) / "megadetector_models").glob("*.pt"),
                       key=lambda p: p.stat().st_mtime)
        new = [p for p in fresh if p not in tmp_models] or fresh[-1:]
        if new:
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new[-1], cache / new[-1].name)
            write_json({mc["model"]: new[-1].name}, cache / "models.json")
            log.info("cached MegaDetector weights in %s", cache)
    em.save(paths.energy / "MegaDetector_infer.json", model=mc["model"], n_images=len(test),
            ms_per_image=1000 * em.wall_s / max(len(test), 1))


if __name__ == "__main__":
    main()
