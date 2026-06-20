# %%
from pathlib import Path
import os

repo = "AnimeFaces256_AEKL"
if Path("utils").exists():
    pass
elif Path(repo + "/utils").exists():
    os.chdir(repo)

# %%

# makes routes

from pathlib import Path

base_dir = Path("Results_AEKL")

for subdir in ["samples", "checkpoints", "logs"]:
    (base_dir / subdir).mkdir(parents=True, exist_ok=True)

sample_path = base_dir / "samples"
checkpoint_path = base_dir / "checkpoints"
log_path = base_dir / "logs"
data_dir = Path(os.getenv("DATA_DIR", "../data"))


# %%
from utils.dataset import (
    load_dataset_from_local_or_hub,
    create_train_test_split,
    Denormalize,
    to_image_range,
)

from utils.training_utils import (
    print_training_progress,
    show_metrics,
    UpdateEma,
    save_checkpoint,
    load_checkpoint,
    visualize_batch,
)
from utils.training_utils import UpdateEma, save_checkpoint, load_checkpoint
from utils.vae_loss import PerceptualMixedKLLoss
from utils.reproducibility import fix_seed
import torch

seed = fix_seed(65535, deterministic=False)

# Load datasets

dataset1 = load_dataset_from_local_or_hub(
    "puruchinera/ad1k-tagged", data_dir, split="train"
)
dataset2 = load_dataset_from_local_or_hub(
    "puruchinera/anime-faces-256px-v2", data_dir, split="train"
)
dataset3 = load_dataset_from_local_or_hub(
    "puruchinera/anime-pixel-art-animefacesv2", data_dir, split="train"
)


train_dataset, test_dataset = create_train_test_split(
    dataset1,
    dataset2,
    dataset3,
    seed=seed,
    test_percentage=0.2,
)
# %%

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if self.transform:
            image = self.transform(image)
        return image


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

sample_size = 256
bs = 32


train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


de_normalize = Denormalize(mean=mean, std=std)

train_image_dataset = ImageDataset(train_dataset, transform=train_transform)
test_image_dataset = ImageDataset(test_dataset, transform=test_transform)


cpus = os.cpu_count() or 1
train_dataloader = DataLoader(
    train_image_dataset,
    batch_size=bs,
    shuffle=True,
    num_workers=cpus,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

test_dataloader = DataLoader(
    test_image_dataset,
    batch_size=bs * 2,
    shuffle=False,
    num_workers=cpus,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

x = next(iter(train_dataloader))
print("Stadistics of the first batch:")
print(f"Min: {x.min().item():.4f}")
print(f"Max: {x.max().item():.4f}")
print(f"Mean: {x.mean().item():.4f}")
print(f"Std: {x.std().item():.4f}")

# %%

visualize_batch(
    x[:8], de_normalize, sample_path, file_name="train_batch_visualization.png"
)


# %%
def visualize_samples(
    model,
    dataloader,
    denormalize,
    num_samples=16,
    file_name="reconstruction_visualization.png",
    interpolation_file_name="interpolation_visualization.png",
):
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            device = next(model.parameters()).device
            batch = batch.to(device)

            # Un solo forward
            reconstructions, mu, logvar = model(batch)

            # Estadísticas del latent
            print(
                f"μ mean={mu.mean():.4f} | "
                f"μ std={mu.std():.4f} | "
                f"μ min={mu.min():.4f} | "
                f"μ max={mu.max():.4f} | "
                f"logvar mean={logvar.mean():.4f} | "
                f"logvar std={logvar.std():.4f} | "
                f"logvar min={logvar.min():.4f} | "
                f"logvar max={logvar.max():.4f}"
            )

            # Reconstrucciones
            originals = denormalize(batch[:num_samples].cpu())
            reconstructions_vis = denormalize(reconstructions[:num_samples].cpu())

            comparison = torch.stack(
                [img for pair in zip(originals, reconstructions_vis) for img in pair]
            )

            visualize_batch(
                comparison,
                lambda x: x,
                sample_path=sample_path,
                file_name=file_name,
            )

            # Interpolaciones múltiples usando μ
            if batch.shape[0] >= 2:
                steps = 8

                # Usa tantos pares como permita el batch
                num_pairs = min(batch.shape[0] // 2, 8)

                rows = []

                for pair_idx in range(num_pairs):
                    z1 = mu[2 * pair_idx : 2 * pair_idx + 1]
                    z2 = mu[2 * pair_idx + 1 : 2 * pair_idx + 2]

                    row = []

                    for t in torch.linspace(0, 1, steps, device=device):
                        z = torch.lerp(z1, z2, t)

                        decoded = model.decoder(z)

                        if isinstance(decoded, tuple):
                            decoded = decoded[0]

                        row.append(denormalize(decoded[0].cpu()))

                    rows.append(torch.stack(row))

                # [num_pairs, steps, C, H, W]
                interpolation_grid = torch.stack(rows)

                # [num_pairs * steps, C, H, W]
                interpolation_grid = interpolation_grid.flatten(0, 1)
                interpolation_grid = interpolation_grid.squeeze(1)
                visualize_batch(
                    interpolation_grid,
                    lambda x: x,
                    sample_path=sample_path,
                    file_name=interpolation_file_name,
                )

            break


# %%
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def run_step(
    model,
    batch,
    loss_fn,
    accelerator,
    optimizer=None,
    training=False,
    kl_weight=1e-6,
):
    if training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    with torch.set_grad_enabled(training):
        with accelerator.autocast():
            reconstructions, mean, logvar = model(batch)
            loss, loss_components = loss_fn(
                reconstructions, batch, mean, logvar, run_gamma=kl_weight
            )
        if training:
            accelerator.backward(loss)
            optimizer.step()
    metric_reconstructions = to_image_range(reconstructions, de_normalize)
    metric_targets = to_image_range(batch, de_normalize)
    psnr = peak_signal_noise_ratio(
        metric_reconstructions, metric_targets, data_range=1.0
    )
    ssim = structural_similarity_index_measure(
        metric_reconstructions, metric_targets, data_range=1.0
    )
    metrics = {
        "loss": loss.detach(),
        "psnr": psnr.detach(),
        "ssim": ssim.detach(),
        "kl": loss_components["kl"],
        "lpips": loss_components["lpips"],
        "reconstruction": loss_components["reconstruction"],
    }
    return metrics


def evaluate(model, dataloader, loss_fn, accelerator, kl_weight=1e-6):
    model.eval()
    all_metrics = {
        "loss": [],
        "psnr": [],
        "ssim": [],
        "kl": [],
        "lpips": [],
        "reconstruction": [],
    }
    with torch.no_grad():
        for batch in dataloader:
            metrics = run_step(
                model,
                batch,
                loss_fn,
                accelerator,
                training=False,
                kl_weight=kl_weight,
            )
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    averaged_metrics = {
        key: torch.stack(values).mean() for key, values in all_metrics.items()
    }
    return averaged_metrics


# %%

from Models.AutoEncoder_KL import VAE
from accelerate import Accelerator

model = VAE()
ema_model = VAE()
ema_model.load_state_dict(model.state_dict())

# %%
accelerator = Accelerator(mixed_precision="bf16")


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
loss_fn = PerceptualMixedKLLoss(alpha=1, beta=0.5, gamma=1e-6).to(accelerator.device)
start_epoch = 0
resume_step_in_epoch = 0
global_step = 0
best_val_loss = float("inf")

save_every_steps = 500

last_checkpoint_dir = checkpoint_path / "last_checkpoint"
best_checkpoint_dir = checkpoint_path / "best_checkpoint"

if last_checkpoint_dir.exists():
    print("Loading last checkpoint...")

    (
        start_epoch,
        resume_step_in_epoch,
        best_val_loss,
        global_step,
    ) = load_checkpoint(
        last_checkpoint_dir,
        model,
        ema_model,
        optimizer,
        scheduler=None,
    )

    accelerator.print(
        f"Resuming from epoch={start_epoch} "
        f"step={resume_step_in_epoch} "
        f"global_step={global_step}"
    )

model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, test_dataloader
)

ema_model = ema_model.to(accelerator.device)
# %%
num_epochs = 100


metrics = {}
from tqdm import tqdm
import numpy as np


for epoch in range(start_epoch, num_epochs):
    if epoch >= 74:
        optimizer.param_groups[0]["lr"] = 3e-5
        if epoch >= 85:
            optimizer.param_groups[0]["lr"] = 5e-6

    model.train()

    for step, batch in enumerate(
        tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    ):
        metrics = run_step(
            model,
            batch,
            loss_fn,
            accelerator,
            optimizer=optimizer,
            training=True,
        )
        UpdateEma(model, ema_model, bs=batch.shape[0])
        global_step += 1
    print_training_progress(
        epoch,
        num_epochs,
        True,
        metrics,
        log_path,
        file_name=f"training_metrics.txt",
    )

    save_checkpoint(
        last_checkpoint_dir,
        accelerator.unwrap_model(model),
        accelerator.unwrap_model(ema_model),
        optimizer,
        scheduler=None,
        epoch=epoch + 1,
        step_in_epoch=step + 1,
        best_val_loss=best_val_loss,
        global_step=global_step,
    )
    accelerator.print(f"Checkpoint saved at step {global_step} and epoch {epoch + 1}")

    ema_model.eval()
    val_metrics = evaluate(ema_model, test_dataloader, loss_fn, accelerator)

    print_training_progress(
        epoch,
        num_epochs,
        False,
        val_metrics,
        log_path,
        file_name=f"validation_metrics.txt",
    )

    visualize_samples(
        ema_model,
        test_dataloader,
        de_normalize,
        num_samples=16,
        file_name=f"reconstruction_epoch_{epoch + 1}.png",
    )

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"].item()
        save_checkpoint(
            best_checkpoint_dir,
            accelerator.unwrap_model(model),
            accelerator.unwrap_model(ema_model),
            optimizer,
            scheduler=None,
            epoch=epoch + 1,
            step_in_epoch=0,
            best_val_loss=best_val_loss,
            global_step=global_step,
        )
        accelerator.print(f"New best model saved with val_loss={best_val_loss:.4f}")
