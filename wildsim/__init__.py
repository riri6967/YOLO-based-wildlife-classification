"""wildsim: simulate how detector uncertainty and environmental conditions
propagate into ecological monitoring outcomes for camera-trap surveys."""

import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.json"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        return json.load(f)


def logit(p):
    import numpy as np
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expit(x):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-x))
