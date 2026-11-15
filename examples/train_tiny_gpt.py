"""Train a tiny character-level GPT on sample text (educational example only).

This toy model learns patterns in the short text below; it is not suitable as a
ChatGPT replacement and is not connected to the NVGPT Pages app.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters and sample data
batch_size = 16
block_size = 32
max_iters = 500
learning_rate = 1e-3
eval_interval = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_size = 64
num_heads = 4
num_layers = 3

text = """
Artificial intelligence is transforming how humans interact with computers.
By learning statistical patterns from vast amounts of text, language models
can predict the next word in a sentence, answer questions, write code,
and assist with creative tasks. Transformers rely on self-attention to
weigh the importance of every word relative to all other words in context.
""" * 20

# Character-level tokenizer
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(token_ids):
    return "".join(itos[i] for i in token_ids)


data = torch.tensor(encode(text), dtype=torch.long)


def get_batch():
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scaled dot-product attention uses the head dimension, not embed_size.
        head_size = k.shape[-1]
        weights = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        return weights @ self.value(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList(
            [SelfAttentionHead(head_size) for _ in range(num_heads)]
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
        normalized = self.ln1(x)
        attended = torch.cat([head(normalized) for head in self.heads], dim=-1)
        x = x + self.proj(attended)
        return x + self.ffwd(self.ln2(x))


class CompleteGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        if T > block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {block_size}")
        tok_emb = self.token_embedding(idx)
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(positions)
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_final(x))

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = CompleteGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Training on {device}...")
for step in range(max_iters):
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == max_iters - 1:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")

print("\n--- Generated Output ---")
start_prompt = torch.tensor([encode("Artificial")], dtype=torch.long, device=device)
generated_ids = model.generate(start_prompt, max_new_tokens=150)
print(decode(generated_ids[0].tolist()))
