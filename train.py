from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.losses import RateDistortionLoss
from hcg_rvq.models import build_model
from hcg_rvq.utils import load_config, seed_everything


def configure_optimizers(model: torch.nn.Module, config: dict):
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 1e-4)
    aux_lr = train_cfg.get("aux_lr", 1e-3)

    parameters = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and not name.endswith(".quantiles")
    }
    aux_parameters = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and name.endswith(".quantiles")
    }

    params = [param for name, param in model.named_parameters() if name in parameters]
    aux_params = [param for name, param in model.named_parameters() if name in aux_parameters]

    optimizer = torch.optim.Adam(params, lr=lr)
    aux_optimizer = torch.optim.Adam(aux_params, lr=aux_lr) if aux_params else None
    return optimizer, aux_optimizer


def maybe_init_wandb(config: dict, run_dir: Path):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    import wandb

    return wandb.init(
        project=wandb_cfg.get("project", "HCG-RVQ"),
        name=wandb_cfg.get("name"),
        mode=wandb_cfg.get("mode", "online"),
        config=config,
        dir=str(run_dir),
    )


def to_float_dict(values: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer, aux_optimizer, epoch: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 1234))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = config.get("run_name", Path(args.config).stem)
    run_dir = Path(config.get("output_dir", "experiments")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    dataset = ImageFolderDataset(
        roots=data_cfg.get("train_roots", ["/dpl/openimages"]),
        patch_size=data_cfg.get("patch_size", 256),
        training=True,
        max_images=data_cfg.get("max_train_images"),
    )
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=device == "cuda",
        drop_last=True,
    )

    model = build_model(config).to(device)
    criterion = RateDistortionLoss(**config.get("loss", {}))
    optimizer, aux_optimizer = configure_optimizers(model, config)

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if aux_optimizer is not None and checkpoint.get("aux_optimizer") is not None:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)

    wandb_run = maybe_init_wandb(config, run_dir)
    epochs = config.get("train", {}).get("epochs", 1)
    log_interval = config.get("train", {}).get("log_interval", 50)
    clip_max_norm = config.get("train", {}).get("clip_max_norm")

    for epoch in range(start_epoch, epochs):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in progress:
            x = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()

            output = model(x)
            losses = criterion(output, x)
            losses["loss"].backward()
            if clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            aux_loss = torch.tensor(0.0, device=device)
            if aux_optimizer is not None and hasattr(model, "aux_loss"):
                aux_loss = model.aux_loss()
                aux_loss.backward()
                aux_optimizer.step()

            log_values = to_float_dict(losses)
            log_values["aux_loss"] = float(aux_loss.detach().cpu())
            for key, value in output.get("rvq_stats", {}).items():
                if torch.is_tensor(value):
                    log_values[f"rvq/{key}"] = float(value.detach().cpu())

            progress.set_postfix({k: f"{v:.4f}" for k, v in log_values.items() if k in {"loss", "bpp_total", "psnr"}})
            if wandb_run is not None and global_step % log_interval == 0:
                wandb_run.log(log_values, step=global_step)
            global_step += 1

        save_checkpoint(run_dir / "checkpoint_latest.pth.tar", model, optimizer, aux_optimizer, epoch, global_step)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

