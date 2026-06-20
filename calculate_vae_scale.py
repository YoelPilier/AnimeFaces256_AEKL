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
)

from utils.reproducibility import fix_seed
import torch
import datasets

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


dataset = datasets.concatenate_datasets([dataset1, dataset2, dataset3])

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
bs = 128


transform = transforms.Compose(
    [
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


image_dataset = ImageDataset(dataset, transform=transform)


cpus = os.cpu_count() or 1
dataloader = DataLoader(
    image_dataset,
    batch_size=bs,
    shuffle=True,
    num_workers=cpus,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)


from Models.AutoEncoder_KL import VAE
from accelerate import Accelerator
from safetensors.torch import load_file
from tqdm import tqdm

model = VAE()

loaded_state = load_file(checkpoint_path / "last_checkpoint/ema_model.safetensors")
model.load_state_dict(loaded_state)
# %%
accelerator = Accelerator(mixed_precision="bf16")
model, dataloader = accelerator.prepare(model, dataloader)

total_sq_sum = 0
total_sum = 0
total_count = 0

model.eval()
for batch in tqdm(dataloader, desc="Calculating mean and std"):
    with torch.no_grad():
        with accelerator.autocast():
            mu, logvar = model.encoder(batch)
        total_sum += mu.sum()
        total_sq_sum += (mu**2).sum()
        total_count += mu.numel()

mean = total_sum / total_count
var = total_sq_sum / total_count - mean**2
std = torch.sqrt(var)

scale_factor = 1.0 / std

print(f"Scale factor: {scale_factor.item()}")

with open(log_path / "vae_scale_factor.txt", "w") as f:
    f.write(f"Scale factor: {scale_factor.item()}\n")
