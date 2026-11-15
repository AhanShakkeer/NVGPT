import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    """One head of causal self-attention; tokens can only look backward."""

    def __init__(self, embed_size, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)

        # Scale by the per-head dimension, as in scaled dot-product attention.
        head_size = k.shape[-1]
        weights = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)

        v = self.value(x)
        return weights @ v


class TransformerBlock(nn.Module):
    """A Transformer layer combining attention and feed-forward computation."""

    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList(
            [SelfAttentionHead(embed_size, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_size, embed_size)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_out = torch.cat([head(self.ln1(x)) for head in self.heads], dim=-1)
        x = x + self.proj(attn_out)
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """A small GPT-style Transformer blueprint for educational use."""

    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=4, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_size, num_heads, block_size) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx):
        _, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Predict and append one token at a time."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
