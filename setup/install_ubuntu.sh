#!/usr/bin/env bash
# Set up the pipeline on Ubuntu 24.04 (the base of ROS 2 Jazzy) with an NVIDIA GPU.
#
#   bash setup/install_ubuntu.sh
#
# Creates two virtualenvs in the repo:
#   .venv     training / evaluation / simulation (PyTorch, Ultralytics, timm, ...)
#   .venv-md  MegaDetector (kept separate because its dependencies differ)
set -euo pipefail
cd "$(dirname "$0")/.."

# A sourced ROS environment puts /opt/ros/jazzy packages on PYTHONPATH, which can
# shadow pip packages (numpy, opencv). Keep this project isolated from it.
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH || true

echo "== system packages (needs sudo)"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-pip build-essential git wget unzip libgl1 libglib2.0-0

echo "== GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "WARNING: nvidia-smi not found. Install the NVIDIA driver first, e.g.:"
  echo "    sudo ubuntu-drivers install && sudo reboot"
  echo "Training on CPU is possible for a smoke test but far too slow for the real dataset."
fi

PY=python3
echo "== main environment (.venv)"
$PY -m venv .venv
.venv/bin/pip install --upgrade pip wheel
# PyPI torch wheels bundle CUDA 12 runtime libraries; only the NVIDIA driver is needed.
.venv/bin/pip install torch torchvision
.venv/bin/pip install -r requirements-pipeline.txt

echo "== MegaDetector environment (.venv-md)"
$PY -m venv .venv-md
.venv-md/bin/pip install --upgrade pip wheel
.venv-md/bin/pip install megadetector

echo "== check"
.venv/bin/python - <<'EOF'
import torch, ultralytics, timm
print("torch", torch.__version__, "CUDA available:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("ultralytics", ultralytics.__version__, "timm", timm.__version__)
EOF
.venv-md/bin/python -c "import megadetector; print('megadetector OK')"
echo
echo "Done. Next:  bash run_all.sh            (everything)"
echo "       or:   bash run_all.sh --quick    (small trial run, ~3k images, few epochs)"
