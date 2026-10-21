"""Train the species classifier used after MegaDetector (two-stage pipeline).

Trains a timm backbone on ground-truth crops with a frequency-tempered
sampler (to help rare species) and random greyscale augmentation (to
mimic night-time IR frames). Selects the epoch with the best macro-F1 on
the validation crops.

    python -m nkpipe.train_classifier
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from . import cli, load_classes, log, write_json
from .energy import EnergyMeter

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def eval_transform(size):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def build_model(arch, n_classes, pretrained):
    import timm
    return timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)


class CropFolder(datasets.ImageFolder):
    """ImageFolder whose sub-folders are class indices ("0", "1", ...), not names."""

    def find_classes(self, directory):
        import os
        dirs = sorted((d for d in os.listdir(directory) if d.isdigit()), key=int)
        return dirs, {d: int(d) for d in dirs}


def macro_f1(y, p, k):
    f1 = []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        if tp + fn == 0:
            continue
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1)) if f1 else 0.0


def main():
    args, cfg, paths = cli(__doc__, lambda ap: ap.add_argument(
        "--force", action="store_true", help="train again even if a trained classifier exists"))
    if (paths.models / "classifier.json").exists() and not args.force:
        log.info("species classifier already trained; skipping. Use --force to retrain.")
        return
    cc = cfg["train"]["classifier"]
    names = load_classes(paths)
    k = len(names)
    dev = torch.device("cpu" if cfg["train"]["device"] == "cpu" or not torch.cuda.is_available()
                       else f"cuda:{cfg['train']['device']}")

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(cc["img_size"], scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.02),
        transforms.RandomGrayscale(0.25),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = CropFolder(paths.crops / "train", tf_train)
    val_ds = CropFolder(paths.crops / "val", eval_transform(cc["img_size"]))

    counts = np.bincount(train_ds.targets, minlength=k).astype(float)
    w = (np.maximum(counts, 1) ** -cc["balance_power"])[train_ds.targets]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
    dl_train = DataLoader(train_ds, cc["batch"], sampler=sampler, num_workers=cc["workers"], pin_memory=True,
                          drop_last=True)
    dl_val = DataLoader(val_ds, cc["batch"] * 2, num_workers=cc["workers"], pin_memory=True)

    model = build_model(cc["arch"], k, cc["pretrained"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cc["lr"], weight_decay=cc["weight_decay"])
    steps = cc["epochs"] * len(dl_train)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cc["lr"], total_steps=max(steps, 1), pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=cc["label_smoothing"])
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out = paths.models / "classifier.pt"
    best = -1.0

    gpu = None if dev.type == "cpu" else dev.index
    with EnergyMeter(device_index=gpu) as em:
        for ep in range(cc["epochs"]):
            model.train()
            tot = 0.0
            for x, y in dl_train:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast(dev.type, enabled=use_amp):
                    loss = crit(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item() * len(y)
            model.eval()
            ys, ps = [], []
            with torch.no_grad(), torch.autocast(dev.type, enabled=use_amp):
                for x, y in dl_val:
                    ps.append(model(x.to(dev)).argmax(1).cpu())
                    ys.append(y)
            y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
            f1, acc = macro_f1(y, p, k), float((y == p).mean())
            log.info("epoch %d/%d loss %.3f  val acc %.3f  macro-F1 %.3f", ep + 1, cc["epochs"],
                     tot / max(len(train_ds), 1), acc, f1)
            if f1 > best:
                best = f1
                torch.save({"arch": cc["arch"], "names": names, "img_size": cc["img_size"],
                            "state_dict": model.state_dict()}, out)
    em.save(paths.energy / "classifier_train.json", model="species_classifier", weights=str(out),
            best_val_macro_f1=best)
    write_json({"model": "classifier", "weights": str(out), "best_val_macro_f1": best},
               paths.models / "classifier.json")


if __name__ == "__main__":
    main()
