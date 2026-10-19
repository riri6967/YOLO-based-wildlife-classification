"""Per-image environmental covariates computed from pixels and EXIF.

* ir / night  -- greyscale infrared frames (all channels ~equal) or a night-time timestamp
* green_frac  -- share of vegetation pixels (excess-green index) in daytime frames;
                 aggregated per camera site into a habitat vegetation-density score
* contrast, sharpness -- low values flag rain, fog, lens moisture or motion blur
                 (a proxy; label a subset by hand to validate it)
"""

from datetime import datetime

import numpy as np
from PIL import Image

EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME = 306
EXIF_IFD = 0x8769


def _exif_datetime(img):
    try:
        exif = img.getexif()
        v = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
        return datetime.strptime(str(v).strip("\x00 "), "%Y:%m:%d %H:%M:%S").isoformat() if v else None
    except Exception:
        return None


def _laplacian_var(g):
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def image_features(path, ir_threshold=4.0):
    out = {"width": None, "height": None, "exif_datetime": None, "chroma": np.nan,
           "ir": np.nan, "green_frac": np.nan, "contrast": np.nan, "sharpness": np.nan,
           "readable": False}
    try:
        img = Image.open(path)
        out["width"], out["height"] = img.size
        out["exif_datetime"] = _exif_datetime(img)
        img.draft("RGB", (img.size[0] // 8, img.size[1] // 8))  # fast JPEG downscale
        img = img.convert("RGB")
        img.thumbnail((320, 320))
        a = np.asarray(img, dtype=np.float32)
    except Exception:
        return out
    h = a.shape[0]
    a = a[int(0.08 * h): int(0.92 * h)]  # drop the info bars most camera traps burn in
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    chroma = float((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    s = r + g + b + 1e-6
    exg = 2 * g / s - r / s - b / s
    is_ir = chroma < ir_threshold
    out.update(chroma=chroma, ir=float(is_ir),
               green_frac=np.nan if is_ir else float((exg > 0.05).mean()),
               contrast=float(gray.std() / 255.0), sharpness=_laplacian_var(gray / 255.0),
               readable=True)
    return out


def _features_star(args):
    return image_features(*args)
