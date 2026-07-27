# ML Deep-Dive Summer Plan

Goal: close the gap between "masters-level basics" and "how modern systems like ChatGPT/Claude actually work" — through small from-scratch implementations, not just reading.

Structure: 5 tracks. Each has a **build target** (something runnable), **core papers** (read closely, not skimmed), and a **checklist**. Do them roughly in order, but Track 0 can run in parallel with anything since it's your foundation refresher.

---

## Track 0 — Foundations Refresher (keep, don't skip)
You already started this. Keep it lightweight — a working implementation each, no need to gold-plate.

**Build targets:**
- [ ] MLP + backprop from scratch (numpy only, no autograd)
- [ ] A tiny autograd engine (Karpathy's `micrograd` style) — this is the single best exercise for truly understanding backprop
- [ ] Basic CNN (ResNet block) on CIFAR-10/MNIST
- [ ] Vanilla Transformer encoder-decoder (translation or copy task)
- [ ] Vanilla VAE on MNIST

**Papers:**
- He et al., *Deep Residual Learning for Image Recognition* (ResNet, 2015)
- Vaswani et al., *Attention Is All You Need* (2017)
- Kingma & Welling, *Auto-Encoding Variational Bayes* (VAE, 2013)

---

## Track 1 — Modern LLM Internals
The architectural deltas between "vanilla transformer" and "what GPT-4/Claude/Llama actually use."

**Build target:** A small GPT (nanoGPT-style, ~10-50M params) trained on a toy dataset (e.g. TinyStories or Shakespeare), incorporating:
- RoPE instead of learned/sinusoidal positional embeddings
- RMSNorm instead of LayerNorm
- SwiGLU activation instead of ReLU/GELU MLP
- Grouped-Query Attention (GQA) instead of vanilla multi-head attention
- KV-caching for inference

**Papers:**
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE, 2021)
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (RMSNorm, 2019)
- Shazeer, *GLU Variants Improve Transformer* (SwiGLU, 2020)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023)
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models* (2023) — good reference implementation combining all of the above
- Optional: Fedus et al., *Switch Transformers* (MoE, 2021) — if you want to also add a mixture-of-experts layer

**Reference code:** Karpathy's `nanoGPT` and `nanochat` repos — read before writing your own, don't copy blind.

---

## Track 2 — Post-Training (the biggest conceptual gap)
This is genuinely what a masters curriculum usually skips entirely, and it's most of "why does ChatGPT feel like ChatGPT" (as opposed to a raw pretrained model).

**Build target:**
- [ ] SFT (supervised fine-tuning) on your Track 1 GPT using an instruction dataset
- [ ] DPO (Direct Preference Optimization) on top of the SFT model using a small preference dataset
- Optional stretch: implement PPO-based RLHF to see why DPO was such a simplification

**Papers:**
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT/RLHF, 2022) — read this even if you skip implementing PPO, it's the conceptual anchor
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023)
- Schulman et al., *Proximal Policy Optimization Algorithms* (PPO, 2017) — only if doing the stretch goal

---

## Track 3 — Vision-Language Models (VLMs)
Turns out to be "transformer + vision encoder + a small connector," which is reassuring once you've built one.

**Build target:**
- [ ] CLIP-style contrastive image-text model on a small paired dataset (Flickr8k or similar)
- [ ] Minimal LLaVA-style setup: freeze a small vision encoder + your Track 1 GPT, train a small projection layer to align them

**Papers:**
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, 2021)
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT, 2020)
- Liu et al., *Visual Instruction Tuning* (LLaVA, 2023)

---

## Track 4 — Pick ONE niche deep-dive
Choose based on interest, don't try to do both this summer.

### Option A: RL for Reasoning
**Build target:** GRPO (or simplified PPO) on a small verifiable-reward task (e.g. arithmetic or a toy logic puzzle) with your Track 1 GPT.

**Papers:**
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO, 2024)
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)
- Optional: Ouyang et al. (InstructGPT, already in Track 2) as the RLHF baseline to compare against

### Option B: Geometric Deep Learning / GNNs beyond the basics
**Build target:** An equivariant GNN (e.g. E(n)-equivariant) on a toy molecular property prediction task.

**Papers:**
- Bronstein et al., *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* (2021) — the map of the whole territory, read this first
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (GCN, 2016) — baseline if your masters didn't already cover this in depth
- Satorras et al., *E(n) Equivariant Graph Neural Networks* (2021)

---

## Generative Models (folded in, not a separate track — but worth having)
Since generative modeling underlies a lot of "new" stuff (diffusion, image gen), a small side build is worth it:
- [ ] DDPM-style diffusion model on MNIST/CIFAR (small U-Net)

**Papers:**
- Ho et al., *Denoising Diffusion Probabilistic Models* (DDPM, 2020)
- Song et al., *Denoising Diffusion Implicit Models* (DDIM, 2020) — faster sampling, good follow-up

---

## Suggested pacing (rough, adjust freely)
| Weeks | Track |
|---|---|
| 1-2 | Track 0 (finish what you started) |
| 3-4 | Track 1 (modern GPT internals) |
| 5 | Generative side-quest (diffusion) |
| 6-7 | Track 2 (post-training: SFT + DPO) |
| 8-9 | Track 3 (VLM) |
| 10+ | Track 4 (your choice: RL reasoning or geometric DL) |

## Repo structure suggestion
```
ml-summer/
  PLAN.md              <- this file, update checkboxes as you go
  00_foundations/
  01_modern_gpt/
  02_post_training/
  03_vlm/
  04_niche/            <- rl_reasoning/ or geometric/
  side_quests/
    diffusion/
  papers/              <- pdfs or notes per paper, organized by track
```

## Working with Claude Code
Start each coding session by pointing Claude Code at this file plus the specific track you're working on, e.g.:
> "Read PLAN.md, I'm working on Track 1 today — help me implement RoPE in my attention layer."

Come back to this chat (or a new claude.ai chat) for conceptual questions, debugging design decisions, or re-planning — use Claude Code for the actual implementation/debugging loop.