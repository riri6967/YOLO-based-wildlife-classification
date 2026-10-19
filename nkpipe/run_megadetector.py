"""Run MegaDetector (stage 1 of the two-stage pipeline) on the test split.

MegaDetector lives in its own virtualenv (.venv-md, see setup/install_ubuntu.sh)
because its dependencies differ from Ultralytics'. This wrapper writes the image
list, calls MegaDetector's batch runner in that environment, and measures GPU energy.

    python -m nkpipe.run_megadetector
"""

import subprocess

from . import ROOT, cli, log, write_json
from .energy import EnergyMeter
from . import preds as P


def main():
    args, cfg, paths = cli(__doc__)
    mc = cfg["megadetector"]
    test = P.test_images(paths)
    img_list = paths.preds / "md_input.json"
    write_json([str(p) for p in test.path], img_list)
    out = paths.preds / "megadetector_raw.json"
    py = mc["python"] if mc["python"].startswith("/") else str(ROOT / mc["python"])
    cmd = [py, "-m", "megadetector.detection.run_detector_batch", mc["model"], str(img_list), str(out),
           "--threshold", str(mc["threshold"]), *map(str, mc.get("extra_args", []))]
    log.info("running: %s", " ".join(cmd))
    device = cfg["train"]["device"]
    with EnergyMeter(device_index=None if device == "cpu" else str(device)) as em:
        subprocess.run(cmd, check=True)
    em.save(paths.energy / "MegaDetector_infer.json", model=mc["model"], n_images=len(test),
            ms_per_image=1000 * em.wall_s / max(len(test), 1))


if __name__ == "__main__":
    main()
