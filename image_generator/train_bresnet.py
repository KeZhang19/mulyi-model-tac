"""Train BResNet for paired tactile image regression.

Example:
    python image_generator/train_bresnet.py \
        --data-dir image_generator/output_5000 \
        --direction rgb2depth \
        --epochs 80 \
        --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from image_generator.bresnet import BResNet
except ImportError:
    from bresnet import BResNet


@dataclass
class SplitCfg:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class PairedTactileDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], direction: str, augment: bool = False) -> None:
        self.pairs = pairs
        self.direction = direction
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tacmap_path, rgb_path = self.pairs[idx]
        if self.direction == "rgb2depth":
            x = load_rgb_tensor(rgb_path)
            y = load_depth_tensor(tacmap_path)
        else:
            x = load_rgb_tensor(tacmap_path)
            y = load_rgb_tensor(rgb_path)
        if self.augment:
            x, y = augment_pair(x, y)
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "output_5000"))
    parser.add_argument("--run-dir", type=str, default=str(Path(__file__).resolve().parent / "runs" / "bresnet_rgb_depth"))
    parser.add_argument("--direction", choices=("rgb2depth", "tacmap2rgb"), default="rgb2depth")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-test-preds", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(Path(args.data_dir))
    if not pairs:
        raise FileNotFoundError(f"No paired files found under {args.data_dir}/tacmap and {args.data_dir}/rgb")

    split_cfg = SplitCfg(float(args.train_ratio), float(args.val_ratio), float(args.test_ratio))
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, split_cfg, int(args.seed))
    save_splits(run_dir / "splits.json", train_pairs, val_pairs, test_pairs)
    print(f"[data] pairs={len(pairs)} train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}", flush=True)
    print(f"[task] direction={args.direction}", flush=True)

    train_loader = DataLoader(
        PairedTactileDataset(train_pairs, direction=str(args.direction), augment=not bool(args.no_augment)),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        PairedTactileDataset(val_pairs, direction=str(args.direction)),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
    )
    test_loader = DataLoader(
        PairedTactileDataset(test_pairs, direction=str(args.direction)),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
    )

    device = torch.device(str(args.device))
    print(f"[device] training on {device}", flush=True)
    out_channels = 1 if args.direction == "rgb2depth" else 3
    model = BResNet(in_channels=3, out_channels=out_channels, base_channels=int(args.base_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", best_val))

    history: list[dict] = []
    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, train=True)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None, scaler=None, train=False)
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **prefix("train", train_metrics), **prefix("val", val_metrics)}
        history.append(row)
        append_jsonl(run_dir / "metrics.jsonl", row)
        print(
            f"[epoch {epoch:03d}] train_l1={row['train_l1']:.5f} val_l1={row['val_l1']:.5f} "
            f"val_mse={row['val_mse']:.5f}",
            flush=True,
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / "last.pt")
        if val_metrics["l1"] < best_val:
            best_val = val_metrics["l1"]
            ckpt["best_val"] = best_val
            torch.save(ckpt, run_dir / "best.pt")

    best = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_metrics = run_epoch(model, test_loader, device, optimizer=None, scaler=None, train=False)
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    save_test_predictions(model, test_pairs, str(args.direction), device, run_dir / "test_triplets", int(args.save_test_preds))
    print(f"[test] l1={test_metrics['l1']:.5f} mse={test_metrics['mse']:.5f} psnr={test_metrics['psnr']:.2f}dB", flush=True)


def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    tacmap_dir = data_dir / "tacmap"
    rgb_dir = data_dir / "rgb"
    tacmaps = {p.stem: p for p in tacmap_dir.glob("*.png")}
    rgbs = {p.stem: p for p in rgb_dir.glob("*.png")}
    stems = sorted(tacmaps.keys() & rgbs.keys())
    return [(tacmaps[s], rgbs[s]) for s in stems]


def split_pairs(
    pairs: list[tuple[Path, Path]],
    cfg: SplitCfg,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if min(cfg.train, cfg.val, cfg.test) < 0.0:
        raise ValueError("Split ratios must be non-negative.")
    total = cfg.train + cfg.val + cfg.test
    if total <= 0.0:
        raise ValueError("At least one split ratio must be positive.")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    train_end = int(len(indices) * cfg.train / total)
    val_end = train_end + int(len(indices) * cfg.val / total)
    shuffled = [pairs[i] for i in indices]
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def save_splits(path: Path, train_pairs, val_pairs, test_pairs) -> None:
    payload = {
        "train": [p[0].stem for p in train_pairs],
        "val": [p[0].stem for p in val_pairs],
        "test": [p[0].stem for p in test_pairs],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_rgb_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_depth_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def augment_pair(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        x = torch.flip(x, dims=(2,))
        y = torch.flip(y, dims=(2,))
    if random.random() < 0.5:
        x = torch.flip(x, dims=(1,))
        y = torch.flip(y, dims=(1,))
    return x, y


def run_epoch(model, loader, device, optimizer, scaler, train: bool) -> dict[str, float]:
    model.train(train)
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    l1_sum = 0.0
    mse_sum = 0.0
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(scaler is not None and scaler.is_enabled())):
                pred = torch.sigmoid(model(x))
                l1 = l1_loss(pred, y)
                mse = mse_loss(pred, y)
                loss = l1 + 0.25 * mse
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch = int(x.shape[0])
            l1_sum += float(l1.detach().cpu()) * batch
            mse_sum += float(mse.detach().cpu()) * batch
            count += batch
    l1_mean = l1_sum / max(1, count)
    mse_mean = mse_sum / max(1, count)
    psnr = -10.0 * np.log10(max(mse_mean, 1.0e-12))
    return {"l1": l1_mean, "mse": mse_mean, "psnr": float(psnr)}


def save_test_predictions(model, pairs, direction: str, device, out_dir: Path, limit: int) -> None:
    if limit <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for _i, (tacmap_path, rgb_path) in enumerate(pairs[:limit]):
            if direction == "rgb2depth":
                x_path = rgb_path
                target_path = tacmap_path
                x = load_rgb_tensor(x_path).unsqueeze(0).to(device)
                input_img = Image.open(x_path).convert("RGB")
                target = Image.open(target_path).convert("L").convert("RGB")
            else:
                x_path = tacmap_path
                target_path = rgb_path
                x = load_rgb_tensor(x_path).unsqueeze(0).to(device)
                input_img = Image.open(x_path).convert("RGB")
                target = Image.open(target_path).convert("RGB")
            pred = torch.sigmoid(model(x))[0].detach().cpu()
            pred_img = tensor_to_image(pred)
            make_triplet(input_img, pred_img, target).save(out_dir / f"{x_path.stem}_input_pred_target.png")


def make_triplet(input_img: Image.Image, pred_img: Image.Image, target_img: Image.Image) -> Image.Image:
    w, h = input_img.size
    gap = 8
    canvas = Image.new("RGB", (w * 3 + gap * 2, h), (32, 32, 32))
    canvas.paste(input_img, (0, 0))
    canvas.paste(pred_img, (w + gap, 0))
    canvas.paste(target_img, ((w + gap) * 2, 0))
    return canvas


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    x = x.clamp(0.0, 1.0)
    if x.shape[0] == 1:
        arr = (x[0].numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prefix(name: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{name}_{k}": v for k, v in values.items()}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
