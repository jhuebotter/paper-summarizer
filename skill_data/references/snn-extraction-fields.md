# SNN Control Extraction Fields — Definitions & Guidance

**Every field: 1 sentence (one line). Learning mechanism: 2 sentences max. Notable findings: bullet points.  
Numbers: neuron/parameter counts and DOF only. No hyperparameters. No padding.**

**Filling rules (must follow):**
- No blanks: use `not reported` / `not applicable` / `not applicable (survey)` as appropriate.
- Any specific numeric value or concrete claim should include a short anchor: `Source: Sec. X / Fig. Y / Tbl. Z / App. A`.
- Prefer compact phrasing with semicolons/parentheses over multiple sentences.

---

## Global rules

**"not applicable" vs. "not reported":**
- `not applicable` — concept genuinely doesn't apply (e.g. "online vs. offline" for a system with no learning)
- `not reported` — concept applies but paper omits it; this is a transparency gap worth noting
- `not applicable (survey)` — survey paper; details are aggregated, not per-system

**Opinion / commentary papers:** Describe the target system with attribution "(as described in [author year], re [target paper])". Do not describe the commentary author's perspective arguments as technical properties of the system.

**Strength of evidence:** If a field is filled based on what a paper claims rather than demonstrates, note this (e.g., “claimed”, “measured”, “estimated”). "Authors claim energy efficiency of X" differs from "measured energy on Loihi: X J/inference."

**Vocabulary (canonical tokens):** Use `fully spiking` vs `hybrid` where applicable; inference hardware must be one of `CPU/GPU`, `Neuromorphic emulator/SDK`, `Physical neuromorphic chip`.

---

## Neuron model
The type of spiking (or rate-coded) neuron (e.g., LIF / ALIF / Izhikevich / rate-coded / Hodgkin-Huxley / custom); note if the choice is justified/ablated vs framework default, and key state variables only if explicitly reported. 

---

## Network architecture
Feedforward / recurrent / reservoir, etc.; list any non-spiking modules (readout, bottleneck, critic), and whether it is `fully spiking` or `hybrid` (explicitly state `hybrid` if any non-spiking component participates in the learning/control loop).

**Fully spiking vs. hybrid:** A spiking actor with a non-spiking DNN critic is `hybrid` — this is common in spiking RL and undermines whole-system energy claims.

---

## Model scale
Total neurons and/or parameters only; note whether it is proof-of-concept scale vs benchmark-relevant scale (qualitatively).

---

## Simulator / framework
Be specific — "PyTorch" alone is insufficient if a spiking library is used on top (e.g., SNNTorch / BindsNet / Control Stork / Nengo+NengoDL / Brian2 / Lava / SpikingJelly / Norse / custom JAX/TF/NumPy / analogue hardware only / `not reported`).

---

## Hardware (training)
GPU (model if reported) / CPU / neuromorphic chip / `not reported`.

---

## Controller hardware (inference)
Where does the network execute at runtime? Use exactly one:
- **CPU/GPU** — standard compute
- **Neuromorphic emulator/SDK** — chip behaviour simulated in software (e.g. NengoLoihi emulator, Lava software stack); no physical chip present
- **Physical neuromorphic chip** — real hardware (e.g. Loihi board, SpiNNaker rack); required for energy measurements to be meaningful

**Do not conflate with the task environment.** A controller can run on a real Loihi chip while the robot arm is simulated in MuJoCo — these are orthogonal axes reported separately.

**Energy (if discussed):** state one of `Measured` / `Estimated` / `Claimed` / `not reported` and the evaluation platform (chip vs emulator vs CPU/GPU); write “claimed but not measured” if no numbers are given.

Neuromorphic chip examples (non-exhaustive): Loihi/Loihi 2 (Intel), BrainScaleS/BrainScaleS-2 (Heidelberg), SpiNNaker/SpiNNaker2 (Manchester), TrueNorth (IBM), Dynap-SE (ETH), Tianjic (Tsinghua).

---

## Control task
What is controlled, in what environment, toward what goal (be concrete; include task name if given).

---

## Task type
Discrete action (which?) / continuous torque and/or velocity / mixed.

---

## Task complexity & scale
Required: state DOF and action DOF; optionally (if explicitly evaluated) note partial observability/noise/varying targets and whether robustness/transfer is tested; only compare to “typical benchmarks” if the paper explicitly positions itself that way.

---

## Simulation environment
Name the environment and whether it is simulated or real; common options: MuJoCo / Isaac Sim / Isaac Lab / PyBullet / Gazebo / Webots / OpenAI Gym / Gymnasium / DMControl / ALE / PyGame / custom / real robot / `not reported`.

---

## Spike encoding
How continuous inputs are converted into spikes or currents (common options): Rate/Poisson coding; population coding; temporal/latency coding; direct current injection; learned linear projection into currents; delta/event-based; `not reported`; note if encoding is ablated/justified vs framework default.

---

## Action decoding
How outputs are decoded (population vector; membrane potential readout; firing rate averaging; weighted spike count; NEF decoding; `not reported`); if decoding uses non-spiking readout or rate windowing, note that outputs are not strictly event-driven.

---

## Learning mechanism
The most important field: name the algorithm, what signal drives learning (loss/reward/modulator), what is local vs global, and any hybrid or multi-phase structure; must include **(1) algorithm family/name, (2) training signal/loss, (3) optimization structure (end-to-end vs staged)**; see `learning-paradigms.md`.

---

## Credit assignment scope
- **Global** — BPTT-style gradients (truncated/full as stated) propagate through trained components
- **Semi-local** — eligibility traces + modulating signal (e-prop, FPTT, perturbation); avoids full BPTT storage
- **Local** — pre/post-synaptic activity only (STDP, R-STDP, three-factor rules)
- **Analytical** — no gradient; weights set by closed-form optimization (NEF, reservoir)
- **Hybrid** — combines scopes; describe each component separately
- **Not applicable** — no learning: "Not applicable — weights [analytically constructed / fixed by design]."

---

## Online vs. offline
Online (weights update during task execution) / offline (separate training phase from dataset or replay) / mixed; do not confuse online inference with online learning.

---

## Data collection
Self-generated via environment interaction / pre-collected dataset / teacher forcing from reference model / replay buffer / synthetic data from learned forward model / `not reported`.

---

## Key training details
Structural decisions only: multi-phase setup, curriculum, ANN pretraining stage, online-to-offline transition, initialization strategy; no hyperparameter values (learning rates, batch sizes, time constants).

---

## Comparison to baselines
Baselines used and key claimed advantage; note:
- **Missing baselines:** which of (1) non-spiking ANN on same task, (2) classical controller (PID/LQR/MPC), (3) prior SNN method, are absent?
- **Statistical rigor:** averaged over multiple seeds with variance vs single-run; flag: "single-run result — no variance reported."
- **Task comparability:** same conditions for all baselines, or differences that make comparison ambiguous?
- **Parity check (minimal):** are observation/action spaces, training budget, and evaluation conditions comparable across baselines (as stated), or mismatched/unclear?