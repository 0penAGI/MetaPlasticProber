# MetaPlastic Prober

A research system built on top of autoregressive language models (default: GPT-2).

Not a wrapper. Not a fine-tune. A second nervous system layered over the first — one that remembers not what was said, but how the state moved.

-----

## The problem it addresses

Standard transformers are memoryless between tokens in the way that matters: each hidden state is a function of input, not of trajectory. The system forgets how it got here.

MetaPlastic Prober adds temporal depth — multiple interacting timescales that track not just the current state but its derivatives, its drift, its history of deformation.

-----

## Architecture

### Three temporal layers

**Fast memory** — token-coupled, reactive. Decays rapidly. Tracks immediate context.

**Slow fields** (`slow_base`, `slow_vel`, `slow_attr`) — trajectory memory. Updated holographically: the past deforms rather than overwrites. Isolated per prompt to prevent cross-contamination.

**Meta-dynamics** — measures deviation of state derivatives (`dS`), orthogonalizes hidden evolution against its own history. Prevents the system from collapsing into the nearest attractor.

### Key mechanisms

- Per-prompt slow memory isolation (separate banks, no global collapse)
- Gated holographic slow updates (frozen during early exploration phase)
- Closed feedback loop: `logits → hidden dynamics → logits` (non-Markovian hidden space)
- Temperature-coupled phase noise injection
- Soft amplitude gating instead of hard clamp
- Dynamic temperature `T_dyn` tracking hidden variance in real time

-----

## What it measures

The core diagnostic is **trajectory divergence**: how differently does the hidden state evolve from two semantically distinct prompts?

A system with no memory converges — both prompts collapse to the same representation quickly. A system with trajectory memory holds the divergence.

-----

## Experimental results

Tested on GPT-2 (117M). Same base model, same generation loop, same temperature.

**Semantic sensitivity (cosine divergence at step 0):**

|Mode                   |Divergence|
|-----------------------|----------|
|Without meta-plasticity|0.026     |
|With meta-plasticity   |0.457     |

**17x increase in semantic sensitivity.** The architecture distinguishes between *“The most beautiful thing about consciousness is”* and *“Violence emerges when memory collapses into fear”* — the base model nearly cannot.

**Entropy (average over 60 generation steps):**

|Mode        |Avg entropy|
|------------|-----------|
|Without meta|2.85       |
|With meta   |3.10       |

**Order parameter** (hidden state variance):

|Mode        |Variance|
|------------|--------|
|Without meta|0.004   |
|With meta   |0.006   |

Higher variance = richer, less collapsed hidden dynamics.

**Slow memory drift:** 66° → 83° over generation. Memory rotates — it does not collapse.

-----

## What the numbers mean

The base model treats two semantically opposite prompts as nearly identical in hidden space (divergence 0.026). The meta-plastic layer opens that gap to 0.457 — the system begins to feel the difference.

This is not improved accuracy. It is improved sensitivity to meaning over time.

-----

## Limitations

Convergence still occurs by step 5 on GPT-2. This is a property of the base model’s representation space, not of the meta-plastic layer. On larger bases (Mistral, Gemma) the trajectory separation is expected to hold longer.

GPT-2 is used here as a fast experimental proxy.

-----

## Research directions

- Nonlinear memory in neural networks
- Temporal stability of semantic representations
- Emergence-like dynamics in recurrent transformer extensions
- Consciousness-adjacent architectures: what does it mean for a system to remember its own movement?

-----

## Run

```bash
pip install torch transformers numpy
python meta_plastic_probing.py
```

Requires MPS (Apple Silicon) or CUDA. Falls back to CPU.

-----

*This system does not converge to answers. It converges to trajectories.*
