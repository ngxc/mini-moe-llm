import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast

class InferenceConfig:
    vocab_path = "bert-base-chinese"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 512
    hidden_dim = 512
    n_layers = 24
    n_heads = 8
    dropout = 0.1
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    checkpoint = r"E:\deep-learning\ml\moe_pre\new_pt.pt"
    num_experts = 2
    top_k = 1
    moe_hidden_ratio = 2

cfg = InferenceConfig()

class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=2, top_k=1, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim * 2),
                nn.GLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        gate_logits = self.gate(x)
        gate_scores = F.softmax(gate_logits, dim=-1)

        topk_val, topk_idx = torch.topk(gate_scores, self.top_k, dim=-1)
        topk_val = topk_val / (topk_val.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(x)

        for i in range(self.top_k):
            expert_idx = topk_idx[..., i]
            expert_weight = topk_val[..., i].unsqueeze(-1)

            for e in range(self.num_experts):
                mask = (expert_idx == e).float().unsqueeze(-1)
                if mask.sum() == 0:
                    continue
                expert_out = self.experts[e](x)
                expert_out = self.output_proj(expert_out)
                out += expert_out * expert_weight * mask

        return out

class MiniLLMBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1, num_experts=2, top_k=1, moe_hidden_ratio=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden_dim = dim * moe_hidden_ratio
        self.ff = MoEFeedForward(dim, hidden_dim, num_experts=num_experts, top_k=top_k, dropout=dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                          attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask)[0]
        x = x + self.ff(self.ln2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_len=512, dropout=0.1,
                 num_experts=2, top_k=1, moe_hidden_ratio=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([
            MiniLLMBlock(hidden_dim, n_heads, dropout,
                          num_experts=num_experts, top_k=top_k, moe_hidden_ratio=moe_hidden_ratio)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None, labels=None):
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None

        for blk in self.layers:
            x = blk(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.ln(x)
        logits = self.head(x)
        return logits

@torch.no_grad()
def top_k_top_p_sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature is None or temperature <= 1e-8:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float('-inf')), logits)

    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        cutoff = cum_probs > top_p
        cutoff[..., 0] = False
        sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_id

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str = cfg.device,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token: str = "<|im_end|>",
):
    model.eval()
    eos_id = tokenizer.convert_tokens_to_ids(eos_token) if eos_token is not None else getattr(tokenizer, "eos_token_id", None)

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False, max_length=cfg.max_len)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= cfg.max_len:
            break

        logits = model(input_ids, attention_mask=attention_mask)[:, -1, :].squeeze(0)
        next_id = top_k_top_p_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        next_id = next_id.view(1, 1)

        input_ids = torch.cat([input_ids, next_id.to(device)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_id, device=device)], dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text

def load_tokenizer_and_model():
    tokenizer = BertTokenizerFast.from_pretrained(cfg.vocab_path)
    tokenizer.add_special_tokens({"additional_special_tokens": cfg.special_tokens})
    vocab_size = len(tokenizer)

    model = MiniLLM(
        vocab_size=vocab_size,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_len=cfg.max_len,
        dropout=cfg.dropout
    ).to(cfg.device)

    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        state = torch.load(cfg.checkpoint, map_location=cfg.device)
        model.load_state_dict(state, strict=False)

    model.head.weight = model.embed.weight
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = load_tokenizer_and_model()
    prompt = "<|im_start|>给我讲一个关于狗的趣闻。<|im_end|> <|im_start|>"
    out = generate_text(
        model, tokenizer, prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_k=90,
        top_p=0.85
    )
    print(out)
