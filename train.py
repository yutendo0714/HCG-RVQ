from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hcg_rvq.data import ImageFolderDataset
from hcg_rvq.losses import RateDistortionLoss
from hcg_rvq.models import build_model
from hcg_rvq.utils import load_config, load_matching_state_dict, seed_everything


def has_freeze_policy(config: dict) -> bool:
    train_cfg = config.get("train", {})
    return bool(train_cfg.get("freeze_prefixes") or train_cfg.get("freeze_schedule"))


def freeze_prefixes_for_step(config: dict, step: int) -> tuple[str, ...]:
    train_cfg = config.get("train", {})
    schedule = train_cfg.get("freeze_schedule")
    if not schedule:
        return tuple(train_cfg.get("freeze_prefixes", []))

    for phase in schedule:
        until_step = phase.get("until_step")
        if until_step is None or step < int(until_step):
            return tuple(phase.get("freeze_prefixes", []))
    return ()




def loss_overrides_for_step(config: dict, step: int) -> dict[str, float]:
    train_cfg = config.get("train", {})
    schedule = train_cfg.get("loss_schedule")
    if not schedule:
        return {}

    for phase in schedule:
        until_step = phase.get("until_step")
        if until_step is None or step < int(until_step):
            return dict(phase.get("loss_overrides", {}) or {})
    return {}


def apply_loss_overrides(criterion: torch.nn.Module, overrides: dict[str, float]) -> None:
    if not overrides:
        return
    for name, value in overrides.items():
        if not hasattr(criterion, name):
            raise AttributeError(f"loss criterion has no configurable field: {name}")
        setattr(criterion, name, float(value))

def apply_freeze_prefixes(model: torch.nn.Module, freeze_prefixes: tuple[str, ...]) -> None:
    frozen = 0
    frozen_tensors = 0
    trainable = 0
    trainable_tensors = 0
    for name, param in model.named_parameters():
        should_freeze = any(name.startswith(prefix) for prefix in freeze_prefixes)
        param.requires_grad_(not should_freeze)
        if should_freeze:
            frozen += param.numel()
            frozen_tensors += 1
        else:
            trainable += param.numel()
            trainable_tensors += 1
    print(
        "Freeze policy active: "
        f"frozen {frozen_tensors} tensors / {frozen} parameters, "
        f"trainable {trainable_tensors} tensors / {trainable} parameters, "
        f"prefixes={freeze_prefixes}"
    )


def lr_multiplier_for_name(name: str, multipliers: dict[str, float]) -> float:
    best_prefix_len = -1
    best_multiplier = 1.0
    for prefix, multiplier in multipliers.items():
        if name.startswith(prefix) and len(prefix) > best_prefix_len:
            best_prefix_len = len(prefix)
            best_multiplier = float(multiplier)
    return best_multiplier


def configure_optimizers(model: torch.nn.Module, config: dict):
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 1e-4)
    aux_lr = train_cfg.get("aux_lr", 1e-3)
    include_frozen = has_freeze_policy(config)
    lr_multipliers = train_cfg.get("param_lr_multipliers", {}) or {}

    parameters = {
        name
        for name, param in model.named_parameters()
        if (include_frozen or param.requires_grad) and not name.endswith(".quantiles")
    }
    aux_parameters = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and name.endswith(".quantiles")
    }

    grouped_params: dict[float, list[torch.nn.Parameter]] = defaultdict(list)
    grouped_names: dict[float, list[str]] = defaultdict(list)
    for name, param in model.named_parameters():
        if name not in parameters:
            continue
        multiplier = lr_multiplier_for_name(name, lr_multipliers)
        grouped_params[multiplier].append(param)
        grouped_names[multiplier].append(name)

    param_groups = []
    for multiplier in sorted(grouped_params, reverse=True):
        param_groups.append(
            {
                "params": grouped_params[multiplier],
                "lr": lr * multiplier,
                "lr_multiplier": multiplier,
                "name_count": len(grouped_names[multiplier]),
            }
        )
        print(
            f"Optimizer group: lr={lr * multiplier:g}, multiplier={multiplier:g}, "
            f"tensors={len(grouped_names[multiplier])}"
        )

    aux_params = [param for name, param in model.named_parameters() if name in aux_parameters]

    optimizer = torch.optim.Adam(param_groups, lr=lr)
    aux_optimizer = torch.optim.Adam(aux_params, lr=aux_lr) if aux_params else None

    scheduler = None
    milestones = train_cfg.get("lr_milestones") or train_cfg.get("lr_epochs")
    if milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(m) for m in milestones],
            gamma=train_cfg.get("lr_gamma", 0.1),
        )
    return optimizer, aux_optimizer, scheduler


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


def anchor_loss_active(criterion: torch.nn.Module) -> bool:
    return any(
        float(getattr(criterion, name, 0.0)) != 0.0
        for name in (
            "rho_anchor_mu",
            "rho_anchor_log_s",
            "rho_anchor_u",
            "rho_anchor_y_hat",
            "rho_anchor_selected_distortion_margin",
        )
    )


def build_anchor_model(config: dict, device: str) -> torch.nn.Module | None:
    train_cfg = config.get("train", {})
    anchor_path = train_cfg.get("anchor_model")
    if not anchor_path:
        return None

    anchor_config_path = train_cfg.get("anchor_config")
    anchor_config = load_config(anchor_config_path) if anchor_config_path else config
    anchor_model = build_model(anchor_config).to(device)
    checkpoint = torch.load(anchor_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing_keys, unexpected_keys = anchor_model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(
            "Anchor load non-strict: "
            f"missing={list(missing_keys)}, unexpected={list(unexpected_keys)}"
        )
    anchor_model.eval()
    for param in anchor_model.parameters():
        param.requires_grad_(False)
    if anchor_config_path:
        print(f"Loaded anchor model from {anchor_path} with config {anchor_config_path}")
    else:
        print(f"Loaded anchor model from {anchor_path}")
    return anchor_model


def to_float_dict(values: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def load_teacher_labels(
    config: dict,
) -> tuple[str, dict[str, float], dict[str, dict[str, float]]] | tuple[None, None, dict[str, dict[str, float]]]:
    label_cfg = config.get("train", {}).get("teacher_labels")
    if not label_cfg:
        return None, None, {}
    path = Path(label_cfg["path"])
    path_column = label_cfg.get("path_column", "path")
    target_column = label_cfg.get("target_column", "householder_reliability_keep")
    target_name = label_cfg.get("target_name", "householder_reliability_keep")
    weight_column = label_cfg.get("weight_column")
    weight_name = label_cfg.get("weight_name", weight_column)
    seed_column = label_cfg.get("seed_column")
    seed_value = label_cfg.get("seed_value")
    if seed_value is None and label_cfg.get("filter_current_seed", False):
        seed_value = config.get("seed")
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    labels: dict[str, float] = {}
    extra_labels: dict[str, dict[str, float]] = {}
    if weight_column:
        extra_labels[str(weight_name)] = {}
    skipped_seed = 0
    duplicates = 0
    for row in rows:
        if seed_column and seed_value is not None and str(row.get(seed_column)) != str(seed_value):
            skipped_seed += 1
            continue
        image_path = str(row[path_column])
        if image_path in labels:
            duplicates += 1
        labels[image_path] = float(row[target_column])
        if weight_column:
            extra_labels[str(weight_name)][image_path] = float(row[weight_column])
    if not labels:
        raise ValueError(f"teacher label file has no rows: {path}")
    extra = ""
    if seed_column and seed_value is not None:
        extra += f", seed_filter={seed_column}=={seed_value}, skipped_seed={skipped_seed}"
    if duplicates:
        extra += f", duplicate_paths_overwritten={duplicates}"
    if weight_column:
        extra += f", weight={weight_name}"
    print(f"Loaded {len(labels)} teacher labels from {path} as target {target_name}{extra}")
    return str(target_name), labels, extra_labels


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer,
    aux_optimizer,
    scheduler,
    epoch: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-model", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 1234))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = config.get("run_name", Path(args.config).stem)
    run_dir = Path(config.get("output_dir", "experiments")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    teacher_target_name, teacher_label_map, teacher_extra_label_maps = load_teacher_labels(config)

    data_cfg = config.get("data", {})
    dataset = ImageFolderDataset(
        roots=data_cfg.get("train_roots", ["/dpl/openimages/open-images-v6/train/data"]),
        patch_size=data_cfg.get("patch_size", 256),
        training=True,
        max_images=data_cfg.get("max_train_images"),
        start_index=data_cfg.get("start_index", 0),
        return_path=bool(teacher_label_map) or bool(data_cfg.get("return_path", False)),
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
    anchor_model = build_anchor_model(config, device)
    init_model = args.init_model or config.get("train", {}).get("init_model")
    if args.resume and init_model:
        raise ValueError("use either --resume for full training state or --init-model for weights, not both")
    if init_model:
        skip_prefixes = tuple(config.get("train", {}).get("init_skip_prefixes", []))
        load_stats = load_matching_state_dict(
            model,
            init_model,
            map_location=device,
            skip_prefixes=skip_prefixes,
        )
        print(f"Initialized weights from {init_model}: {load_stats}")

    start_epoch = 0
    global_step = 0
    active_freeze_prefixes = freeze_prefixes_for_step(config, global_step)
    if active_freeze_prefixes or has_freeze_policy(config):
        apply_freeze_prefixes(model, active_freeze_prefixes)

    criterion = RateDistortionLoss(**config.get("loss", {}))
    active_loss_overrides = loss_overrides_for_step(config, global_step)
    apply_loss_overrides(criterion, active_loss_overrides)
    optimizer, aux_optimizer, scheduler = configure_optimizers(model, config)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if aux_optimizer is not None and checkpoint.get("aux_optimizer") is not None:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)
        active_freeze_prefixes = freeze_prefixes_for_step(config, global_step)
        if active_freeze_prefixes or has_freeze_policy(config):
            apply_freeze_prefixes(model, active_freeze_prefixes)
        active_loss_overrides = loss_overrides_for_step(config, global_step)
        apply_loss_overrides(criterion, active_loss_overrides)

    wandb_run = maybe_init_wandb(config, run_dir)
    train_cfg = config.get("train", {})
    epochs = train_cfg.get("epochs", 1)
    log_interval = train_cfg.get("log_interval", 50)
    clip_max_norm = train_cfg.get("clip_max_norm")
    save_interval = train_cfg.get("save_interval_steps")
    max_steps = train_cfg.get("max_steps")
    stop_training = False

    print(
        f"Training {run_name}: {len(dataset)} images, {len(loader)} steps/epoch, "
        f"epochs={epochs}, max_steps={max_steps}, device={device}"
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in progress:
            next_freeze_prefixes = freeze_prefixes_for_step(config, global_step)
            if next_freeze_prefixes != active_freeze_prefixes:
                active_freeze_prefixes = next_freeze_prefixes
                apply_freeze_prefixes(model, active_freeze_prefixes)
            next_loss_overrides = loss_overrides_for_step(config, global_step)
            if next_loss_overrides != active_loss_overrides:
                active_loss_overrides = next_loss_overrides
                apply_loss_overrides(criterion, active_loss_overrides)
                print(f"Loss overrides active at step {global_step}: {active_loss_overrides}")

            if isinstance(batch, dict):
                x = batch["image"].to(device, non_blocking=True)
                batch_paths = [str(path) for path in batch.get("path", [])]
            else:
                x = batch.to(device, non_blocking=True)
                batch_paths = []
            optimizer.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()

            output = model(x)
            if teacher_label_map is not None:
                if not batch_paths:
                    raise RuntimeError("teacher labels are configured, but the dataloader did not return paths")
                missing = [path for path in batch_paths if path not in teacher_label_map]
                if missing:
                    sample = ", ".join(missing[:3])
                    raise KeyError(f"missing teacher labels for {len(missing)} paths, e.g. {sample}")
                values = [teacher_label_map[path] for path in batch_paths]
                teacher_targets = {
                    teacher_target_name: torch.tensor(values, device=device, dtype=x.dtype)
                }
                for extra_name, extra_map in teacher_extra_label_maps.items():
                    missing_extra = [path for path in batch_paths if path not in extra_map]
                    if missing_extra:
                        sample = ", ".join(missing_extra[:3])
                        raise KeyError(f"missing teacher extra labels for {len(missing_extra)} paths, e.g. {sample}")
                    extra_values = [extra_map[path] for path in batch_paths]
                    teacher_targets[extra_name] = torch.tensor(extra_values, device=device, dtype=x.dtype)
                output["teacher_targets"] = teacher_targets
            if anchor_loss_active(criterion):
                if anchor_model is None:
                    raise RuntimeError("anchor loss is active, but train.anchor_model is not configured")
                with torch.no_grad():
                    anchor_output = anchor_model(x)
                output["anchor_conditioning"] = anchor_output.get("conditioning_tensors", {})
                output["anchor_y_hat"] = anchor_output.get("y_hat")
                output["anchor_x_hat"] = anchor_output.get("x_hat")
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
            if teacher_label_map is not None and "teacher_targets" in output:
                teacher_values = next(iter(output["teacher_targets"].values()))
                log_values["teacher/positive_fraction"] = float(teacher_values.detach().mean().cpu())
            log_values["aux_loss"] = float(aux_loss.detach().cpu())
            log_values["epoch"] = float(epoch)
            log_values["lr"] = float(max(group["lr"] for group in optimizer.param_groups))
            log_values["global_step"] = float(global_step)
            for key, value in active_loss_overrides.items():
                log_values[f"loss_override/{key}"] = float(value)
            for idx, group in enumerate(optimizer.param_groups):
                multiplier = group.get("lr_multiplier", 1.0)
                log_values[f"lr/group_{idx}_mult_{multiplier:g}"] = float(group["lr"])
            for key, value in output.get("rvq_stats", {}).items():
                if torch.is_tensor(value):
                    log_values[f"rvq/{key}"] = float(value.detach().cpu())

            progress.set_postfix(
                {k: f"{v:.4f}" for k, v in log_values.items() if k in {"loss", "bpp_total", "psnr", "lr"}}
            )
            if wandb_run is not None and global_step % log_interval == 0:
                wandb_run.log(log_values, step=global_step)
            global_step += 1

            if save_interval is not None and global_step % int(save_interval) == 0:
                save_checkpoint(
                    run_dir / f"checkpoint_step_{global_step}.pth.tar",
                    model,
                    optimizer,
                    aux_optimizer,
                    scheduler,
                    epoch,
                    global_step,
                )

            if max_steps is not None and global_step >= int(max_steps):
                stop_training = True
                break

        save_checkpoint(
            run_dir / "checkpoint_latest.pth.tar",
            model,
            optimizer,
            aux_optimizer,
            scheduler,
            epoch,
            global_step,
        )
        if scheduler is not None:
            scheduler.step()
        if stop_training:
            break

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
