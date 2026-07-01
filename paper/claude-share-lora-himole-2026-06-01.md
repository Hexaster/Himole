---
source_url: https://claude.ai/share/8686bbb5-2ddf-477b-b127-9a30329ad681
ingested: 2026-06-01
sha256: a9e719ede49a08c0cdf709b73bc07926000d6ee7fd169fd138c16eae6389c3bb
note: Source was an English Socratic teaching conversation (shared by Andrew). Condensed from the Q&A transcript into the technical content; math notation normalized to plain text. A referenced paper attachment ("Files hidden in shared chats") was not accessible in the share.
---

# Understanding LoRA Basics, and the HiMoLE Paper

A Socratic teaching conversation (shared by Andrew) covering LoRA fundamentals,
then extending to the HiMoLE paper (hierarchical mixture of LoRA experts).

## Part 1 — LoRA (Low-Rank Adaptation)

**The problem.** Fine-tuning a giant model (e.g., 70B params) by updating every
weight costs as much memory and compute as training from scratch.

**Key intuition.** When you fine-tune a pretrained model, the *change* in weights
ΔW is surprisingly low-rank — almost all of its information lives in a much
smaller subspace. A 4096×4096 ΔW might need only 8 or 16 effective degrees of
freedom.

**The method.** Instead of learning ΔW directly, approximate it as a product of
two tiny matrices:

    ΔW = A B

If W is 4096×4096, then A is 4096×r and B is r×4096 (e.g., r=8). Only A and B are
trained; the original W is frozen. Trainable parameters drop by ~500×. At
inference: W' = W + AB — same model, same speed, far cheaper to train and store.

**Rank r** is the expressivity-vs-efficiency knob: rank 4 is very cheap, rank 64
gives more capacity at higher cost.

**Why it works.** The fine-tuning task doesn't require rewriting the whole weight
matrix — it requires nudging behavior in a low-dimensional direction in weight
space, and empirically this hypothesis almost always holds.

**Worked example.** For W_Q of shape 4096×4096 with rank r=16: trainable params =
4096×16×2 = 131,072, a ratio of ~0.8% of the 4096×4096 ≈ 16.8M full params.
Small enough to train on a laptop.

**Storage/serving win.** One base model sits on disk once; swap in tiny A,B pairs
(a few MB each) per task — one LoRA for coding, one for medical summarization,
etc.

**Initialization.** ΔW = AB must equal 0 at step 0, so training starts exactly at
the pretrained weights (no corruption of pretrained knowledge). Achieved by
initializing A with random Gaussian and B with all zeros, guaranteeing AB = 0.
(Not both zero — then gradients into A would be zero and nothing would move;
random A gives B immediate useful gradient signal.)

**Why LoRA suits RL fine-tuning (e.g., GRPO):**
1. Massive parameter reduction → fits on modest hardware.
2. Frozen weights anchor against catastrophic forgetting from noisy/volatile RL
   reward signals — updates are confined to the tiny A,B subspace, so the model
   can't drift far from its pretrained state.
3. The reference model needed for the GRPO KL-divergence penalty comes nearly
   for free: it *is* the shared frozen base weights, so memory goes from 2× model
   to ~1× plus two tiny A,B matrices.

## Part 2 — HiMoLE (Hierarchical Mixture of LoRA Experts)

**Motivation.** Vanilla LoRA fine-tunes well on in-distribution (ID) data but
degrades on out-of-distribution (OOD) data (e.g., fine-tune on general biomedical
NER, test on rare-disease NER → big drop).

**Idea.** Replace one LoRA adapter with a **mixture of LoRA experts** plus a
two-level routing strategy. Multiple specialized experts mean OOD data is more
likely to match at least one expert's territory.

**The routing subtlety (Figure 1d).**
- Token-level routing: route on each token's features → *better* ID, *worse* OOD.
  It exploits fine-grained local patterns (e.g., "fever" → symptoms expert) that
  hold on ID data but are brittle on OOD. Low bias, high variance.
- Sentence-level routing: route on whole-sentence meaning → *worse* ID, *better*
  OOD. Coarser but more stable across domains. Higher bias, lower variance.

This is a bias-variance tradeoff in disguise.

**HiMoLE's hierarchical routing.** Do both: sentence-level routing coarsely
assigns a subdomain group, then token-level routing refines within it. Final
gating weight:

    G_hie = TopK( Softmax( G_sen ⊙ G_token ) )

⊙ is element-wise multiplication — the sentence score *gates* the token score, so
an expert gets a high final weight only if both global context and local features
agree (a logical AND). This stabilizes token-level precision with sentence-level
context, preventing brittle token-level overfitting on OOD.

**Experts organization.** Experts are organized as KCGs and KCEs (knowledge
clusters / groups and experts).

**Two-stage training** (Table 3: removing it causes up to 31.9% F1 drop).
Training all experts jointly from random init fails: the router needs to learn
which expert to send each input to, but if all experts are random noise they are
equally bad, so the router gets no useful gradient — a chicken-and-egg deadlock
(experts random → router no signal → router random → experts never specialize).
- Stage 1: train each expert independently on a specific subdomain cluster,
  before the router is involved, so experts are genuinely different by Stage 2.
- Stage 2: train routing jointly; a **diversity loss** penalizes experts for
  becoming similar, since gradient updates would otherwise push experts to
  collapse toward the same solution.
