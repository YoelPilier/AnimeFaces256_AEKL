# %%
import torch.nn as nn
import torch.nn.functional as F

# %%


class SelfAttention(nn.Module):
    def __init__(self, dims, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        assert dims % num_heads == 0
        self.num_heads = num_heads
        self.head_size = dims // num_heads
        self.norm = nn.RMSNorm(dims)
        self.qkv = nn.Linear(dims, dims * 3, bias=False)
        self.proj = nn.Linear(dims, dims, bias=False)
        self.dropout = dropout
        self.dropout_proj = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.qkv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        B, C, W, H = x.shape
        N = W * H
        x = x.reshape(B, C, N).transpose(1, 2)
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape((B, N, self.num_heads, self.head_size)).transpose(1, 2)
        k = k.reshape((B, N, self.num_heads, self.head_size)).transpose(1, 2)
        v = v.reshape((B, N, self.num_heads, self.head_size)).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).reshape((B, N, C))
        out = self.proj(out)
        out = self.dropout_proj(out)

        out = out + x
        out = out.transpose(1, 2).reshape((B, C, W, H))

        return out
