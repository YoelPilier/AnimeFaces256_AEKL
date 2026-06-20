# %%
from torch import nn
import torch

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from Models.Layers.VaeBlocks import Encoder, Decoder


class VAE(nn.Module):
    REPO_ID = "puruchinera/AnimeFaces256_AEKL"

    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        latent_dim=4,
        compression_factor=4,
    ):
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

    @classmethod
    def from_pretrained(
        cls,
        checkpoint="last",
        use_ema=True,
        map_location="cpu",
        cache_dir=None,
        strict=True,
        **kwargs,
    ):
        """
        Creates a VAE and loads pretrained weights from Hugging Face.

        Parameters
        ----------
        checkpoint : str
            Either "last" or "best".

        use_ema : bool
            If True, loads ema_model.safetensors instead of model.safetensors.

        map_location : str or torch.device
            Device where the weights are loaded.

        cache_dir : str | None
            Optional Hugging Face cache directory.

        strict : bool
            Passed to load_state_dict().

        **kwargs
            Arguments forwarded to the VAE constructor.
        """

        if checkpoint not in ("last", "best"):
            raise ValueError("checkpoint must be either 'last' or 'best'.")

        filename = "ema_model.safetensors" if use_ema else "model.safetensors"

        weights_path = hf_hub_download(
            repo_id=cls.REPO_ID,
            filename=f"checkpoints/{checkpoint}_checkpoint/{filename}",
            cache_dir=cache_dir,
        )

        model = cls(**kwargs)

        state_dict = load_file(weights_path, device=map_location)
        model.load_state_dict(state_dict, strict=strict)

        return model

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, z_mean, z_log_var

    def reparameterize(self, mean, log_var):
        log_var = torch.tanh(log_var) * 5
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


# %%
