import torch
import lpips
from torch import nn


def kl_divergence(mean, log_var):
    # log_var = torch.clamp(logvar, -10, 10)
    log_var = torch.tanh(log_var) * 5  # Soft clamp
    kl = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    return kl.mean()


class PerceptualMixedKLLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        beta=0.5,
        gamma=0.1,
        loss_fn=None,
        lpips_backbone="vgg",
    ):
        super(PerceptualMixedKLLoss, self).__init__()
        self.loss_fn = loss_fn or nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lpips = lpips.LPIPS(net=lpips_backbone)
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, x, y, mean, logvar, run_gamma=None):
        gamma = self.gamma if run_gamma is None else run_gamma
        recon = torch.clamp(x, -1, 1)
        target = torch.clamp(y, -1, 1)
        lpips_loss = self.lpips(recon.float(), target.float()).mean()
        reconstruction_loss = self.loss_fn(recon, target)
        kl_loss = kl_divergence(mean, logvar)
        total_loss = (
            self.alpha * lpips_loss + self.beta * reconstruction_loss + gamma * kl_loss
        )
        return total_loss, {
            "lpips": lpips_loss.detach(),
            "reconstruction": reconstruction_loss.detach(),
            "kl": kl_loss.detach(),
        }
