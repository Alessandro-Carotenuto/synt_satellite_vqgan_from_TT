import os
import torch
import torch.nn.functional as F
from torch import Tensor
import transformers



def fix_torch_import_issue(kaggle_flag=False):
    base = os.path.dirname(os.path.abspath(__file__))
    utils_file = os.path.join(base, 'taming-transformers/taming/data/utils.py')
    
    with open(utils_file, 'r') as f:
        content = f.read()
    content = content.replace(
        'from torch._six import string_classes',
        'string_classes = str'
    )
    with open(utils_file, 'w') as f:
        f.write(content)
    print("Fixed torch._six import issue!")

# The following function is from the Hugging Face Transformers library and is used for filtering logits during text generation. 
# It can be used in the transformer part of the code to implement top-k and nucleus (top-p) sampling strategies for generating text. 

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


# Inject the real function into transformers module
def fix_inject_top_k_p_filtering():
    transformers.top_k_top_p_filtering = top_k_top_p_filtering
    print("FIX: Real top_k_top_p_filtering function added to transformers!")


def fix_inject_rope():
    """Monkey-patch mingpt classes to support RoPE positional encoding.
    Must be called after taming is importable (i.e. from taming_interface.py).
    Backward-compatible: use_rope=False behaves identically to the original code.
    """
    import math
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import taming.modules.transformer.mingpt as m

    def _build_rope_freqs(seq_len, head_dim, base=10000):
        half = head_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float) / half))
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        return torch.cos(freqs), torch.sin(freqs)

    def _apply_rope(x, cos, sin, start=0):
        B, nh, T, hs = x.shape
        x1, x2 = x[..., :hs//2], x[..., hs//2:]
        c = cos[start:start+T].unsqueeze(0).unsqueeze(0)
        s = sin[start:start+T].unsqueeze(0).unsqueeze(0)
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

    m._build_rope_freqs = _build_rope_freqs
    m._apply_rope = _apply_rope
    m.GPTConfig.use_rope = False

    def _csa_init(self, config):
        nn.Module.__init__(self)
        assert config.n_embd % config.n_head == 0
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.use_rope = getattr(config, 'use_rope', False)
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            cos, sin = _build_rope_freqs(config.block_size, head_dim)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def _csa_forward(self, x, layer_past=None, rope_offset=0):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.use_rope:
            q = _apply_rope(q, self.rope_cos, self.rope_sin, start=rope_offset)
            k = _apply_rope(k, self.rope_cos, self.rope_sin, start=rope_offset)
        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, present

    m.CausalSelfAttention.__init__ = _csa_init
    m.CausalSelfAttention.forward  = _csa_forward

    def _block_forward(self, x, layer_past=None, return_present=False, rope_offset=0):
        if return_present: assert not self.training
        attn, present = self.attn(self.ln1(x), layer_past=layer_past, rope_offset=rope_offset)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x

    m.Block.forward = _block_forward

    def _gpt_init(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                  embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, use_rope=False):
        nn.Module.__init__(self)
        config = m.GPTConfig(vocab_size=vocab_size, block_size=block_size,
                             embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                             n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                             n_unmasked=n_unmasked, use_rope=use_rope)
        self.tok_emb  = nn.Embedding(config.vocab_size, config.n_embd)
        self.use_rope = getattr(config, 'use_rope', False)
        if not self.use_rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop   = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[m.Block(config) for _ in range(config.n_layer)])
        self.ln_f   = nn.LayerNorm(config.n_embd)
        self.head   = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def _gpt_forward(self, idx, embeddings=None, targets=None):
        token_embeddings = self.tok_emb(idx)
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if not self.use_rope:
            x = self.drop(token_embeddings + self.pos_emb[:, :t, :])
        else:
            x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def _gpt_forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None):
        assert not self.training
        token_embeddings = self.tok_emb(idx)
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head,
                              past_length, self.config.n_embd // self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            if not self.use_rope:
                position_embeddings = self.pos_emb[:, past_length, :]
        else:
            if not self.use_rope:
                position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]
        if not self.use_rope:
            x = self.drop(token_embeddings + position_embeddings)
        else:
            x = self.drop(token_embeddings)
        rope_offset = past_length if (past is not None and self.use_rope) else 0
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None,
                               return_present=True, rope_offset=rope_offset)
            presents.append(present)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, torch.stack(presents)

    m.GPT.__init__            = _gpt_init
    m.GPT.forward             = _gpt_forward
    m.GPT.forward_with_past   = _gpt_forward_with_past

    print("FIX: RoPE support injected into mingpt!")


def fix_inject_kv_cache():
    """Monkey-patch mingpt with pre-allocated KV cache inference.
    Adds forward_with_cache to CausalSelfAttention, Block, GPT,
    and sample_with_cache as a module-level function.
    Must be called after fix_inject_rope (taming must be importable).
    """
    import math
    import torch
    import torch.nn.functional as F
    import taming.modules.transformer.mingpt as m

    def _csa_forward_with_cache(self, x, kv_cache, cache_len):
        """kv_cache: (2, B, n_head, max_len, head_dim) — writes K,V in-place, no allocation."""
        B, T, C = x.size()
        hs = C // self.n_head
        k = self.key(x).view(B, T, self.n_head, hs).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, hs).transpose(1, 2)
        if self.use_rope:
            q = m._apply_rope(q, self.rope_cos, self.rope_sin, start=cache_len)
            k = m._apply_rope(k, self.rope_cos, self.rope_sin, start=cache_len)
        kv_cache[0, :, :, cache_len:cache_len + T, :] = k
        kv_cache[1, :, :, cache_len:cache_len + T, :] = v
        new_total = cache_len + T
        K_all = kv_cache[0, :, :, :new_total, :]
        V_all = kv_cache[1, :, :, :new_total, :]
        att = (q @ K_all.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.mask[:, :, cache_len:cache_len + T, :new_total] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ V_all).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def _block_forward_with_cache(self, x, layer_kv_cache, cache_len):
        x = x + self.attn.forward_with_cache(self.ln1(x), layer_kv_cache, cache_len)
        x = x + self.mlp(self.ln2(x))
        return x

    def _gpt_forward_with_cache(self, idx, kv_cache, cache_len, embeddings=None):
        """kv_cache: (n_layer, 2, B, n_head, max_len, head_dim). Returns (logits, new_cache_len)."""
        assert not self.training
        token_embeddings = self.tok_emb(idx)
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        T = token_embeddings.shape[1]
        if not self.use_rope:
            x = self.drop(token_embeddings + self.pos_emb[:, cache_len:cache_len + T, :])
        else:
            x = self.drop(token_embeddings)
        for i, block in enumerate(self.blocks):
            x = block.forward_with_cache(x, kv_cache[i], cache_len)
        x = self.ln_f(x)
        return self.head(x), cache_len + T

    @torch.no_grad()
    def _sample_with_cache(x, model, steps, temperature=1., sample_logits=True,
                           top_k=None, top_p=None, callback=None):
        """Autoregressive sampling with pre-allocated KV cache — no torch.cat per step."""
        assert not model.training
        B = x.shape[0]
        device = x.device
        cfg = model.config
        head_dim = cfg.n_embd // cfg.n_head
        dtype = next(model.parameters()).dtype

        kv_cache = torch.zeros(
            cfg.n_layer, 2, B, cfg.n_head, cfg.block_size, head_dim,
            device=device, dtype=dtype
        )

        cond_len = x.shape[1]
        sample = x

        logits, cache_len = model.forward_with_cache(x, kv_cache, cache_len=0)
        logits = logits[:, -1, :] / temperature

        for n in range(steps):
            if callback is not None:
                callback(n)
            filtered = top_k_top_p_filtering(logits, top_k=top_k or 0, top_p=top_p or 1.0)
            probs = F.softmax(filtered, dim=-1)
            x_new = torch.multinomial(probs, 1) if sample_logits else torch.topk(probs, 1, dim=-1)[1]
            sample = torch.cat([sample, x_new], dim=1)
            if n < steps - 1:
                logits, cache_len = model.forward_with_cache(x_new, kv_cache, cache_len=cache_len)
                logits = logits[:, 0, :] / temperature

        del kv_cache
        return sample[:, cond_len:]

    m.CausalSelfAttention.forward_with_cache = _csa_forward_with_cache
    m.Block.forward_with_cache               = _block_forward_with_cache
    m.GPT.forward_with_cache                 = _gpt_forward_with_cache
    m.sample_with_cache                      = _sample_with_cache

    print("FIX: pre-allocated KV cache injected into mingpt!")


fix_torch_import_issue(kaggle_flag="KAGGLE_KERNEL_RUN_TYPE" in os.environ)
fix_inject_top_k_p_filtering()

