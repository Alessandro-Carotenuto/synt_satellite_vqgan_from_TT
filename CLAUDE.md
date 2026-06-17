# Project Notes — synt_satellite_vqgan_from_TT

## Architecture overview

Two-stage model:
- **Stage 1 (frozen)**: VQ-VAE encoder/decoder (`taming/modules/diffusionmodules/model.py`) — CNN with GroupNorm + Swish
- **Stage 2 (trained)**: Transformer prior (`taming/modules/transformer/mingpt.py`) — minGPT-style, conditioned on ground image tokens, predicts satellite image tokens
- Training script: `train_transformer.py`, supervised with cross-entropy on VQ tokens

---

## Future improvement ideas

### 1. RMSNorm (in `mingpt.py`)
Replace `nn.LayerNorm` in `Block` (lines 102–103) with RMSNorm.
- Eliminates mean calculation → ~15% faster norm operation
- No quality difference, purely a compute micro-optimization
- Low effort, low impact — worth doing if squeezing speed

### 2. SwiGLU activation (in `mingpt.py`)
Replace the FFN `nn.Sequential` in `Block` (lines 105–110):
```
Linear(n_embd, 4*n_embd) → GELU → Linear(4*n_embd, n_embd)
```
with:
```
SwiGLU: Swish(x @ W) ⊙ (x @ V) → Linear(2/3 * 4*n_embd, n_embd)
```
- Adds a 3rd linear projection but improves convergence speed
- Best value if training from scratch; less useful when fine-tuning a converged checkpoint

### 3. RoPE — Rotary Positional Embeddings (in `mingpt.py`)
Replace the learned absolute positional embedding `self.pos_emb` (a fixed `nn.Parameter`) with RoPE applied to Q and K inside `CausalSelfAttention.forward`.
- **Most impactful of the three**: satellite VQ tokens have strong spatial structure; relative position between tokens matters more than absolute position
- Generalizes better to sequence lengths not seen during training
- Requires retraining from scratch (positional embedding shape changes)

### 4. RL fine-tuning phase (after SL convergence)
Second training phase using policy gradient after cross-entropy has plateaued.

**Why**: cross-entropy treats all wrong tokens equally; RL can directly optimize perceptual image quality (SSIM, LPIPS) on the decoded output.

**Loop**:
1. Encode ground image → conditioning tokens (as now)
2. Sample satellite token sequence autoregressively from transformer
3. Decode sampled tokens with frozen VQ decoder → generated satellite image
4. Compute reward vs ground-truth satellite (LPIPS already in repo at `taming/modules/losses/lpips.py`)
5. Policy gradient update on transformer

**Algorithm**: GRPO (Group Relative Policy Optimization, used in DeepSeek-R1)
- Sample N completions per ground image (e.g. N=4–8)
- Rank by reward within the group → use relative advantage (no value network needed)
- Add KL penalty from frozen reference policy (the SL checkpoint) to prevent reward hacking

**Main cost**: RL rollout requires autoregressive sampling (256 steps/image) → ~15–20x slower than supervised forward pass. Use small LR, short RL epochs, and interleave with supervised anchor batches to prevent drift.

### 5. Two-transformer conditioned architecture (coarse-to-fine)
Two sequential transformers, each with a frozen VQGAN backbone:

**Transformer 1** (already trained, 1024 codebook):
```
ground tokens (1024) → satellite tokens coarse (1024)
```

**Transformer 2** (to train, 16384 codebook — Option B):
```
ground tokens (16384) + satellite tokens coarse (1024) + satellite tokens fine (16384, predicted)
```
Sequence: 256 + 256 + 256 = 768 tokens → block_size must be increased from 512 to 768.

**Why Option B (ground re-encoded at 16384)**: the fine details of the ground image are informative for predicting fine satellite details. Re-encoding ground with the 16384 VQGAN gives the second transformer richer conditioning signal at no training cost.

**Training procedure**:
1. Train Transformer 1 to convergence on 1024 (already done)
2. Freeze Transformer 1
3. At training time for Transformer 2: encode ground with 16384 VQGAN, encode satellite with 16384 VQGAN (target), encode satellite with 1024 VQGAN (coarse conditioning — teacher forcing with real coarse tokens)
4. At inference: run Transformer 1 autoregressively → get coarse tokens → feed into Transformer 2 → get fine tokens → decode with 16384 decoder

**Key property**: the two transformers are trained independently and sequentially — no joint training needed.
