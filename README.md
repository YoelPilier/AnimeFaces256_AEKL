# AnimeFaces256-AEKL

This repository contains an AutoencoderKL (AEKL) trained from scratch for latent-space image synthesis experiments on anime face datasets. This model is designed specifically for **latent-space synthesis** rather than direct image generation.

## Model Details

- **Architecture:** AutoencoderKL (AEKL)
- **Input Resolution:** 256×256
- **Latent Space:** `(4, 16, 16)`
- **Components:** Encoder–Decoder with KL divergence regularization
- **Features:** Includes attention mechanisms in latent feature extraction and uses EMA (Exponential Moving Average) weights during training for improved stability.
- **VAE Scale Factor:** Fixed at `0.1298828125` (required for correct latent normalization)
- **Author:** Yoel Pilier ([ORCID](https://orcid.org/0009-0003-9014-3506)) ([GitHub](https://github.com/YoelPilier))

### Intended Purpose
- Latent-space image synthesis research
- KL-regularized autoencoders exploration
- Perceptual learning (LPIPS) evaluation
- Foundation model experimentation
- Attention-based VAE studies

---

## Results (Epoch 99)

### Reconstruction
![Reconstruction Epoch 99](https://huggingface.co/puruchinera/AnimeFaces256_AEKL/resolve/main/samples/reconstruction_epoch_99.png)

### Latent Interpolation
![Latent Interpolation](https://huggingface.co/puruchinera/AnimeFaces256_AEKL/resolve/main/samples/interpolation_visualization.png)

---

## Usage

### Installation & Setup

To train or use the model from the source repository:

```bash
git clone [https://github.com/YoelPilier/AnimeFaces256_AEKL](https://github.com/YoelPilier/AnimeFaces256_AEKL)
cd AnimeFaces256_AEKL
pip install -r requirements.txt
```

### Loading the Model

```python
from Models import VAE
import torch

vae = VAE.from_pretrained(checkpoint="best", use_ema=True)
vae = vae.to("cuda")
vae.eval()
```

### Encoding Images into Latents

```python
with torch.no_grad():
    x = batch_images.to("cuda")
    z_mean, z_log_var = vae.encoder(x)
    z = vae.reparameterize(z_mean, z_log_var)
    latents = z * vae.scaling_factor
```

### Decoding Latents into Images

```python
with torch.no_grad():
    z = latents / vae.scaling_factor
    images = vae.decoder(z)
```

---

## Core VAE Implementation

For reference, the core architecture is structured as follows:

```python
class VAE(nn.Module):
    REPO_ID = "puruchinera/AnimeFaces256_AEKL"

    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=4, compression_factor=4):
        super().__init__()

        self.encoder = Encoder(
            input_dim,
            hidden_dim,
            num_downsamples=compression_factor,
            latent_size=latent_dim,
        )

        self.decoder = Decoder(
            input_dim,
            hidden_dim,
            num_upsamples=compression_factor,
            latent_size=latent_dim,
        )

        self.scaling_factor = 0.1298828125
        self.latent_dim = latent_dim

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

    def reparameterize(self, mean, log_var):
        log_var = torch.tanh(log_var) * 5
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
```

---

## Training Details

To train the model yourself using `accelerate`:

```bash
accelerate launch train_vae.py
```

### Hardware & Time
- **Framework:** PyTorch (trained from scratch)
- **Hardware:** NVIDIA A6000
- **Total Training:** 100 epochs
- **Epoch Duration:** ~1.5 hours
- **Total Training Time:** ~150 hours

### Optimizer & Learning Rate Schedule

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-5
)
```

- **Epoch < 74:** `lr = 3e-4`
- **Epoch ≥ 74:** `lr = 3e-5`
- **Epoch ≥ 85:** `lr = 5e-6`

### Losses Evaluated
- Reconstruction loss
- KL Divergence
- LPIPS perceptual loss

---

## Final Training Metrics (Epoch 99)

| Metric | Training Value | Validation Value |
|--------|----------------|------------------|
| **Loss** | 0.1967 | 0.2022 |
| **PSNR** | 22.1035 | 22.1793 |
| **SSIM** | 0.7837 | 0.7853 |
| **KL Divergence** | 219.7652 | 229.9244 |
| **LPIPS** | 0.1533 | 0.1611 |
| **Reconstruction**| 0.0862 | 0.0814 |

### Raw Logs
- [Training Metrics Log](https://huggingface.co/puruchinera/AnimeFaces256_AEKL/raw/main/logs/training_metrics.txt)
- [Validation Metrics Log](https://huggingface.co/puruchinera/AnimeFaces256_AEKL/raw/main/logs/validation_metrics.txt)

## How to Cite

If you use this repository or its code in your research or projects, please cite it as:

```bibtex
@misc{pilier2026animefaces256aekl,
  author       = {Yoel Pilier},
  title        = {AnimeFaces256-AEKL: AutoencoderKL trained for latent-space image synthesis on anime face datasets},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/YoelPilier/AnimeFaces256_AEKL}}
}
```
