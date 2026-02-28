# Learning Paradigms — Taxonomy & Detail Fields (compact)

Use to correctly characterize the **Learning mechanism** field in Part 2. The goal is **not** to force a paper into a single bucket, but to describe precisely what learning algorithm is used. Paradigms are **non-exhaustive** and **non-exclusive**; many papers are hybrids — describe combinations faithfully.

---

## Workflow (how to classify)

1) **What is trained?** (all weights / readout only / actor / critic / world model / encoder / none)  
2) **Training signal?** (supervised/self-supervised loss / prediction error / reward / neuromodulatory scalar / fitness / closed-form / none)  
3) **Credit assignment?** `Global` (BPTT) / `Semi-local` (eligibility traces + modulatory signal) / `Local` (pre/post ± modulator) / `Analytical` / `Hybrid`  
4) **Structure?** end-to-end vs **staged** (Phase 1→2→…) and what is frozen when  
5) **Regime?** `Offline` / `Online` / `Mixed` (dataset / replay / self-rollouts / teacher forcing)

If unclear: describe the actual update rule/training loop rather than forcing a label.

---

## Vocabulary normalization (use these exact labels when applicable)

**Paradigm family labels (canonical):**
- `Gradient-based (surrogate gradient BPTT)`
- `Gradient-based (online approximation: e-prop/FPTT/OSTL)`
- `ANN-to-SNN conversion`
- `Predictive coding / prediction error learning`
- `Reinforcement learning (model-free)`
- `Reinforcement learning (model-based)`
- `Local plasticity (STDP / R-STDP / three-factor)`
- `Homeostatic / intrinsic plasticity (auxiliary)`
- `Evolutionary / black-box optimization`
- `Analytical / closed-form (NEF / reservoir / control law)`
- `Hybrid / multi-phase` (use in addition to phase labels above)

**Evidence labels (must match Notable findings tags):** `Measured` / `Reported` / `Claimed` / `Attributed`  
**Credit assignment labels:** `Global` / `Semi-local` / `Local` / `Analytical` / `Hybrid` / `Not applicable`  
**Regime labels:** `Offline` / `Online` / `Mixed`

---

## Learning mechanism field: required content + template

Describe what the paper **actually does** — algorithm, loss/signal, update rule/structure — not what the paradigm family is “supposed to” provide.

**Required components (always):**
1) **Algorithm family** (and named algorithm if given)  
2) **Training signal / loss**  
3) **Optimization structure** (end-to-end vs staged; what is frozen)  

**Template (≤2 sentences):**
- S1: `[Algorithm family / named algorithm] trains [component(s)] using [training signal/loss]. (Source: Sec./Fig./Tbl.)`
- S2: `Optimization is [end-to-end | staged: Phase A→B→…], using [Offline/Online/Mixed] updates with [dataset/replay/self-rollouts/teacher forcing]. (Source: Sec./Fig./Tbl.)`

**If multi-phase:** keep S2 as `Phase 1: … ; Phase 2: … ; Phase 3: …` (still ≤2 sentences total).

---

## Attribution & evidence (do not leak claims)

The descriptions below explain what paradigms *are*, not endorsements of performance.

Distinguish clearly:
- what this paper **demonstrates**
- what it **claims** without direct support
- what is **attributed** to cited prior work

**Citation hygiene:** Do not cite a claim as supported by this paper if it only appears via a citation to earlier work; mark it **(Attributed)**.

Common trap: “trained on GPU” + “neuromorphic-ready” ⇒ **Claimed** unless deployed/measured.

---

## Paradigm families (non-exhaustive; multi-label allowed)

| Canonical label | What it is (1 line) | Recognize (keywords / cues) | Record (what must appear in Learning mechanism) | Gotcha (common confusion) |
|---|---|---|---|---|
| Gradient-based (surrogate gradient BPTT) | Surrogate gradients + BPTT through time | surrogate gradient, unroll, (truncated/full) BPTT | which components get gradients; loss type; end-to-end vs staged | “online” sometimes means streaming batches, not online learning |
| Gradient-based (online approximation: e-prop/FPTT/OSTL) | Gradient approximation avoiding full unroll storage | e-prop, FPTT, OSTL, eligibility trace + learning signal | name method; modulatory signal source; online/offline regime | eligibility traces ≠ local plasticity by default |
| ANN-to-SNN conversion | Train ANN → transfer weights to SNN | conversion, weight transfer, threshold balancing, rate coding | conversion pipeline; whether any finetuning exists (and how) | “no training required” ignores ANN training; high-timestep SNNs can kill efficiency |
| Predictive coding / prediction error learning | Learning driven by prediction errors (local or global) | predictive coding, free energy, prediction error units, active inference | local vs global; what error drives updates; role in control | predictive model trained by MSE ≠ predictive coding |
| Reinforcement learning (model-free) | Reward-driven policy learning (PPO/SAC/…) | PPO, SAC, TD3, DDPG, DQN, actor-critic | algorithm name; reward objective; surrogate vs non-surrogate updates | RL + surrogate gradients is common; don’t hide it behind “RL” only |
| Reinforcement learning (model-based) | Learn model + use for planning/rollouts/policy opt | world model, Dreamer, Dyna, imagination, planning | what model is learned; how it’s used; staged vs joint training | “model-based control” may be MPC without RL — describe actual optimization |
| Local plasticity (STDP / R-STDP / three-factor) | Pre/post spike-based rule ± global modulator | STDP, R-STDP, neuromodulation, dopamine, three-factor | rule name; modulator (reward/error); online vs offline | e-prop can look like three-factor; verify backprop absence |
| Homeostatic / intrinsic plasticity (auxiliary) | Adjust excitability to stabilize activity | homeostasis, intrinsic plasticity, threshold adaptation | state it as auxiliary unless it’s the main learning driver | don’t mislabel homeostasis as the main learning mechanism |
| Evolutionary / black-box optimization | Fitness-driven search; no backprop | ES, CMA-ES, neuroevolution, perturbation | optimizer name; fitness; what parameters are optimized | perturbation gradient estimate is still backprop-free |
| Analytical / closed-form (NEF / reservoir / control law) | Weights derived analytically; no learning (or readout only) | NEF, least squares decoder, closed-form, reservoir | explicit “no learning” or “readout-only”; method for readout | reservoir + trained readout is hybrid (analytical + gradient/RL) |

---

## Hybrid / multi-phase methods (required representation)

If the paper is staged, describe **each phase** explicitly (do not hide it in one label):

- `Phase 1: [paradigm] trains [component] using [signal], credit assignment [scope].`
- `Phase 2: …`
- `Phase 3: …`

Suggested (not mandatory): **Primary** = the method responsible for most parameter updates of the deployed policy; **Secondary** = pretraining/conversion/auxiliary adaptation.

---

## Edge cases & pitfalls (short list)

- Predictive **control/model** ≠ predictive **coding** (learning rule matters).
- Online **inference** ≠ online **learning** (check if weights update during deployment).
- Eligibility traces ≠ local plasticity (could be e-prop).
- “Spiking” at inference ≠ “trained as SNN” (conversion pipelines).
- Imitation / behavior cloning is supervised learning on expert actions — name it if present.
- Energy efficiency: treat as **Claimed** unless hardware-measured or clearly estimated with method.

---

## Keyword quick-reference (minimal)

| Keyword / phrase | Likely | Verify |
|---|---|---|
| surrogate gradient / unroll / BPTT | Gradient-based (surrogate BPTT) | which components get gradients; staged vs end-to-end |
| eligibility trace + learning signal | e-prop/FPTT/OSTL **or** three-factor | whether backprop is used; what signal modulates updates |
| replay buffer | RL (often off-policy) or offline training | algorithm (SAC/TD3/DDPG); what is spiking (actor/critic) |
| advantage / actor-critic / policy gradient | RL (model-free) | surrogate-gradient vs other updates |
| world model / Dreamer / imagination | RL (model-based) | planning vs backprop-through-model; frozen vs joint |
| conversion / threshold balancing | ANN-to-SNN conversion | whether finetuning exists; rate coding / timesteps |
| NEF / least squares decoder | Analytical / closed-form | any trained components (readout?) |
| CMA-ES / ES / population | Evolutionary | fitness definition; optimized subset vs full net |

---

## Common overclaims to flag in Critical Assessment (keep short)

- “Matches ANNs” → verify capacity + training budget + identical task setup.  
- “Neuromorphic-ready” → verify emulator vs physical chip vs none.  
- “Online & adaptive” → verify plasticity/updates during deployment, not just training.  
- “Energy efficient” → strongest→weakest: hardware-measured J on chip > estimated (method stated) > qualitative claim.

---

## Examples (2 great, 2 bad)

### Great (clear, complete, multi-label when needed)
1) `Gradient-based (surrogate gradient BPTT) trains the spiking world model using a prediction loss with surrogate gradients. Optimization is staged: Phase 1 world model → freeze → Phase 2 policy trained via rollouts through the model, using Offline updates with replay/self-rollouts. (Source: Sec. 3, Fig. 2)`

2) `Reinforcement learning (model-free) (SAC) trains a spiking actor and non-spiking critic from reward using surrogate-gradient backprop for the spiking components. Optimization is end-to-end with Mixed updates via environment rollouts and a replay buffer. (Source: Sec. 4, Tbl. 1)`

### Bad (too vague / misleading)
1) `Uses RL to train the network for control with a learning rate of 0.05.`  ← Missing algorithm, signal details, structure, regime, and scope, arbitrary hyperparamter value.

2) `Predictive coding trains the controller end-to-end.`  ← Calls it predictive coding without specifying the actual update rule; could just be supervised prediction loss.