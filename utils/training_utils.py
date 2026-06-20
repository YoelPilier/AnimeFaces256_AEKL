import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from pathlib import Path
import torch
from safetensors.torch import save_file, load_file, load_model, save_model


@torch.no_grad()
def UpdateEma(model, ema_model, bs=32, half_life=5000):
    ema_decay = 0.5 ** (bs / half_life)
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.lerp_(param.data, 1 - ema_decay)


def clean_state_dict(model):
    state_dict = model.state_dict()

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}

    return state_dict


def save_checkpoint(
    checkpoint_dir,
    model,
    ema_model,
    optimizer,
    epoch,
    step_in_epoch,
    global_step,
    best_val_loss,
    scheduler=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_tmp = checkpoint_dir / "model.tmp.safetensors"
    ema_tmp = checkpoint_dir / "ema_model.tmp.safetensors"
    state_tmp = checkpoint_dir / "training_state.tmp.pt"

    save_file(clean_state_dict(model), str(model_tmp))
    save_file(clean_state_dict(ema_model), str(ema_tmp))

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        },
        state_tmp,
    )

    model_tmp.replace(checkpoint_dir / "model.safetensors")
    ema_tmp.replace(checkpoint_dir / "ema_model.safetensors")
    state_tmp.replace(checkpoint_dir / "training_state.pt")


def load_checkpoint(
    checkpoint_dir,
    model,
    ema_model,
    optimizer,
    scheduler=None,
):
    checkpoint_dir = Path(checkpoint_dir)

    model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"))
    ema_model.load_state_dict(load_file(checkpoint_dir / "ema_model.safetensors"))

    state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )

    optimizer.load_state_dict(state["optimizer"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])

    return (
        state["epoch"],
        state["step_in_epoch"],
        state["best_val_loss"],
        state["global_step"],
    )


def visualize_batch(
    batch, denormalize, sample_path=None, file_name="batch_visualization.png"
):

    batch = denormalize(batch)
    batch = torch.clamp(batch, 0, 1)
    grid = make_grid(batch, nrow=8)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    if file_name:
        if sample_path is not None:
            plt.savefig(sample_path / file_name, bbox_inches="tight")
        else:
            plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def print_training_progress(
    epoch, num_epochs, training=False, metrics=None, base_dir=None, file_name=None
):
    phase = "Training" if training else "Validation"
    print(f"{phase} Epoch [{epoch}/{num_epochs}]")
    if metrics:
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    if file_name and base_dir:
        with open(base_dir / file_name, "a") as f:
            f.write(f"{phase} Epoch [{epoch}/{num_epochs}] - ")
            if metrics:
                f.write(
                    ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
                )
            f.write("\n")


def show_metrics(metrics, base_dir, file_name="training_metrics.png"):
    plt.figure(figsize=(10, 5))
    for key, values in metrics.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid()
    plt.savefig(base_dir / file_name, bbox_inches="tight")
    plt.close()
