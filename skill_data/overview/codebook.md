# Literature-overview codebook: spiking neural networks for control

Use this codebook to fill a structured record for **one research paper** (section 0 says how to
mark reviews and other documents). Every value must
be supported by the paper itself. Quote the sentence that supports it. Use what the paper reports
about **its own** system, not about cited work.

General rules:
- **Exact values.** Use the option values exactly as written here.
- **Multi-select fields** take every option that applies, and `[]` when none applies.
- **Evidence quotes:** copy one short verbatim sentence or clause (at most about 30 words) from the
  paper. Do not paraphrase or merge fragments. If nothing in the paper supports a value, choose
  `not reported` (or the field's equivalent, unless the field defines a default) and leave the
  quote empty.
- **Follow the definitions, not the paper's words.** The paper's own words ("online", "offline",
  "fully neuromorphic", "robust") don't decide a value; this codebook's definitions do.
- **Main system:** if the paper has several experiments or systems, describe its main SNN
  control system: the one in the title or abstract that does the most control-like task. Ignore
  experiments without an SNN, and side benchmarks. Single-choice fields, the metrics and the
  component list describe that system.
- **Text only.** Use only what the text states. Something stated to be in a supplement counts as
  stated.
- **Evaluation and deployment.** The **evaluation** is the runs whose results the paper reports as
  its outcome: test episodes, the real-robot run, the final benchmark. A component is
  **deployed** if it runs during the evaluation. **Training** is everything done to produce it.
- **Papers without a control task** (classification, benchmarks, theory of learning rules): fill
  in what applies, and use `no control task`, `none`, `not applicable` and `[]` elsewhere.
- **Papers without spiking neurons or event sensors** (e.g. pure impulsive-control theory, an
  ANN-only method): `spiking_roles` = `[]`, and fill the rest for the system studied. For
  `interface`, judge the actuation itself: impulses count as `event-native`.

---

## 0. Paper kind

**`paper_kind`** (single):
- `primary research` — the paper presents its own method, model or experiments.
- `review or survey` — reviews, surveys, tutorials, perspectives.
- `other` — editorials, theses' front matter, non-research documents.

For `review or survey` and `other`, fill the remaining fields as well as possible for the main
system discussed, or with `not applicable` / `[]`; they are excluded from the overview tables.

---

## 1. Spiking role and control setting

**`spiking_roles`** (multi-select): what the spiking neurons do in the studied system.
- `sensing` — spike or event encoding of real sensor signals is part of the studied system. This
  covers an event camera or DVS (even if its events are processed by non-spiking code), SNNs that
  process the event stream of the system's own sensor (live or recorded on the robot),
  event-firing photodetectors, spiking sensor neurons (e.g. stretch receptors), a spiking retina
  or cochlea model. These do **not** count: a Poisson, rate or population encoder of simulator
  state vectors (even if it is learned), and benchmark datasets of recorded events (N-MNIST,
  DVS-Gesture) used for classification.
- `state estimation` — a spiking network estimates or predicts states of the plant or environment:
  an observer, a Kalman-like filter, a forward or world model, ego-motion or optic-flow
  estimation, payload or contact detection. An inverse model whose output is the motor command is
  a `controller`.
- `controller` — spiking neurons compute the control commands from observations, errors or a
  clock: policy, controller, motor network, CPG, reflex circuit. A spiking controller with a
  non-spiking linear readout is still a spiking controller. If the SNN only outputs a state
  estimate that a separate non-spiking controller maps to commands, the role is
  `state estimation`, not `controller`.
- `other` — other spiking parts: a spiking classifier with no control role, a spiking critic or
  training-only network, or a plant emulated with spiking neurons.

**`control_level`** (single), for the whole system that contains the SNN:
- `closed-loop` — the system's commands drive a plant (real, simulated, or analysed
  mathematically) and the plant's state, error or sensory feedback flows back into the system.
- `open-loop actuation` — the system's commands drive a plant, but no plant feedback flows back
  during the run (clock-driven pattern generators, replayed trajectories).
- `perception for control` — the system produces a control-relevant estimate (state, obstacle,
  payload, a slicing signal) but never commands a plant. This includes evaluation on recorded
  robot data.
- `no control task` — no plant and no control-relevant estimate (classification, generic
  benchmarks, learning-rule theory).

**`control_evidence`** (string): a quote supporting `control_level`.

**`plant_setting`** (single): `simulated` / `real` / `simulated and real` / `none`.
- `real` means physical hardware is driven by the system or recorded from it: a robot, a drone,
  a motor rig. A simulated model of a commercial robot is `simulated`, whatever the paper calls
  it.
- `none` covers no plant, and plants that are only analysed mathematically.
- For `perception for control`, give the source of the data (e.g. `real` for recordings from a
  robot, `simulated` for observing a simulated plant).

**`plant_dynamics`** (single):
- `linear` — the controlled system is linear or linearized, e.g. double integrator, LTI benchmark,
  linearized cart-pole.
- `nonlinear` — robot arms, legged robots, drones, pendulum swing-up, MuJoCo/Gym tasks and the
  like.
- `both` — results on linear and nonlinear plants.
- `not applicable` — no plant.

Judge the plant as it is simulated or controlled: a nonlinear plant that is simulated through a
linearization counts as `linear`. If the paper doesn't characterise its plant, use `nonlinear`
for robots, vehicles and other physical mechanisms, and `linear` only for plants it states are
linear. For `perception for control` and
`no control task`, use `not applicable` here and in `plant_model_use`.

**`plant_model_use`** (single): does the designer use equations of the plant?
- `known model used in design` — given plant equations are used to derive the controller or to
  train it. Examples: LQR from the system matrices, SCN or NEF controllers built from the
  dynamics, a computed-torque/Jacobian controller, backpropagation through known kinematics.
- `model learned` — a model of the plant (forward, inverse or world model; system
  identification) is learned and used.
- `model-free` — no plant equations are used or learned. This covers model-free RL, evolution,
  imitation of recorded or teacher behaviour, CPG oscillators, and PID or PD gains tuned by hand
  or by trial.
- `not applicable` — no plant.

**`objectives`** (multi-select): the control objectives the paper evaluates. `[]` if none.
- `regulation` — hold a regulated quantity at a set-point or equilibrium. Examples: balancing,
  hovering, holding a posture, standing up, step changes in the set-point of a speed, angle or
  altitude.
- `tracking` — follow a time-varying reference or trajectory, or move an effector to targets.
  Examples: reaching (including point-to-point moves between targets), trajectory following,
  visual servoing, landing, target shifts during an episode, master-slave synchronization.
- `disturbance rejection` — the paper *explicitly evaluates* recovery from external disturbances
  applied during the run (pushes, wind, added loads, perturbing forces). Observation noise alone
  doesn't count.
- `adaptive control` — the controller's parameters adapt online, during operation, to compensate
  for unknown or changed plant properties, and the paper evaluates this compensation. Examples:
  payload change, a pre-attached unknown load, friction drift, damage, a changed arm, learning an
  unknown plant's inverse mapping online. Learning only a forward model (not the controller), or
  offline training, is not adaptive control.
- `locomotion / pattern generation` — gaits, swimming, CPGs and rhythmic movement (including
  control of a rhythm's amplitude or frequency), and MuJoCo locomotion benchmarks (HalfCheetah,
  Hopper, Ant, Walker).
- `navigation / decision-making` — goal-directed navigation, obstacle or collision avoidance, path
  planning, or discrete decisions in an environment (mazes, Atari-style games, grid worlds).
- `state estimation / prediction` — the evaluated output is an estimate or prediction of the
  plant or its environment (forward models, observers, payload detection, ego-motion, obstacle or
  figure-ground segmentation for a robot).
- `other` — any other control objective, e.g. grasping or manipulation sequences; describe it in
  `task`.

Count only the objectives of the main control experiments, not side validation benchmarks.

**`task`** (string): one line naming the plant and task, with the environment and whether it is
simulated or real. Example: "Simulated 2-link arm reaching (MuJoCo); real 6-DoF Kinova arm".

---

## 2. Components: how each part got its parameters

**`components`** (list): list every network component whose parameter values are designed, solved, learned, searched,
converted or fixed. Merge identical layers; there are typically 1–4 components.
- Parts trained together in one optimization (input encoding layer + SNN + linear readout
  trained end to end) form **one** component, named after the network.
- List a part separately when it is trained or obtained differently, or when it is a non-spiking
  network with hidden layers (a CNN/MLP encoder, an ANN critic).
- Separate training **phases** that use different methods count as separate components (e.g.
  "Phase 1: encoder", "Phase 2: policy").
- Include non-spiking parts that are trained or that do computation (ANN critic, ANN encoder, CNN
  front-end, a linear control layer, decoders).
- **Leave out:**
  - RL target networks, which are copies;
  - input encoders with no parameters;
  - the connectivity *pattern* of a network whose weights are learned, including fixed designed
    weights inside it such as lateral inhibition. A chosen architecture is not a hand-designed
    component.
  - standard low-level blocks outside the authors' controller design: the robot's or simulator's
    joint servo loops, the task's reference or target signals, environment-side PD servos;
  - fixed matrices used only by a learning rule (random feedback or error-projection weights) and
    fixed error-feedback gains used only to drive learning (e.g. in FOLLOW);
  - hand-set output filters or decay constants of the decoding (part of the interface).
- Blocks the authors design as part of their controller do count, even if non-spiking: a heading
  or drive feedback law around a CPG, a trajectory generator inside the proposed model, a
  hand-written rule table that computes the commands.
- Task scripting doesn't count: a scripted gripper-closing step, switching between sub-policies,
  episode resets.
- A plant emulated with spiking neurons is part of the plant, not a component.
- Heads of one network (actor and critic heads on a shared spiking trunk) form one component;
  it is deployed and spiking.
- A non-SNN paper (pure control theory) lists its control law as one `hand-designed`, non-spiking
  `controller` component.

For each component:

**`name`** (string): a short name, e.g. "spiking actor", "ANN critic", "NEF decoders",
"readout", "recurrent reservoir".

**`role`** (single): the component's main function during the evaluation. A model whose output
*is* the command (e.g. an inverse model that drives the motors) is a `controller`.
- `controller` — produces or computes commands;
- `critic / value`;
- `state estimation / world model` — forward, inverse or world models, observers;
- `perception / encoder` — a front-end feeding later stages;
- `readout / decoder` — output weights that turn spikes into values;
- `other` — a classifier, learning-signal network, proxy or teacher network.

**`spiking`** (bool): true if this component consists of spiking neurons. A linear readout of
spikes is not spiking.

**`deployed`** (bool): true if it runs while the system performs the evaluated task. A critic used
only in training is false. A learning-signal network that runs during test-time adaptation is
true. A world model that is only evaluated separately for its prediction accuracy is false.

**`obtained_by`** (single): how its parameter values were obtained.
- `hand-designed` — the values implement a designed computation and are set by hand or from a
  formula, with no fitting to data. Examples: PID gains mapped to weights, a hand-wired CPG or
  reflex circuit, weights from a control law.
- `solved` — computed in one step by a closed-form optimization: least-squares decoders (NEF),
  a spike-coding-network solution from a target linear system, a regression readout. NEF weights
  are `solved` even when the function they implement uses hand-chosen gains (e.g. a PID in
  Nengo).
- `learned` — iteratively updated from data or experience by a learning rule or optimizer.
- `searched` — black-box or evolutionary search over whole networks (GA, ES, CMA-ES,
  neuroevolution, Bayesian optimization, grid search over weights). Gradient-based architecture
  search is `learned` with `backprop / BPTT`.
- `converted` — copied from a trained ANN (ANN-to-SNN conversion, or training with rate neurons
  and then running spiking neurons).
- `random / fixed` — random, constant or relay weights that are never changed and implement no
  designed computation: a reservoir's recurrent weights, fixed random projections, one-to-one
  relays, fixed mossy-fibre wiring, a freely chosen or random SCN decoder.

The next four fields apply only to `learned`, `searched` or `converted` components. Use
`not applicable` for the others.

**`signal`** (single): what defines the desired outcome of the updates.
- `supervised / imitation` — explicit targets: labels, teacher or expert actions, a reference
  trajectory, an output error toward a desired value, a teacher clamping the output neurons, a
  target picked from a downstream loss.
- `reinforcement` — scalar evaluative feedback: reward, return, TD error, fitness, a cost to
  minimise over rollouts. This includes a fitness computed from a tracking error against a
  reference, and a task cost backpropagated through a learned model (e.g. predicted distance to
  target). Use `supervised / imitation` only when per-step targets enter the update directly.
- `self-supervised / system identification` — predicting the plant or the input itself:
  next-state prediction, forward-model error, reconstruction, contrast maximization.
- `unsupervised` — no target or evaluation at all: plain STDP or Hebbian correlation learning,
  homeostasis.
- For `converted` components, use the signal that trained the source network.

**`mechanism`** (single): what the weight update uses.
- `backprop / BPTT` — gradients backpropagated through the network (and through time). This
  includes surrogate gradients, truncated BPTT, standard backprop for non-spiking parts, and
  backpropagation through a model.
- `forward-mode / e-prop` — forward-in-time gradient approximations with eligibility traces and
  neuron-specific learning signals: e-prop, FPTT, OSTL, RTRL approximations, and layer-local
  variants (DRTP, OSTTP, TESS).
- `eligibility + modulator (three-factor)` — local pre/post eligibility times one *global* scalar
  modulator: R-STDP, reward-modulated Hebbian, neuromodulated STDP, node or weight perturbation
  with a global reward. A global on/off gate on Hebbian learning also counts as a modulator.
- `local error (LMS-like)` — a neuron- or output-specific error times presynaptic activity: PES,
  the delta rule, FOLLOW, LMS, FORCE/RLS readout learning, error-gated plasticity (a climbing-fibre
  error per Purkinje cell).
- `Hebbian / STDP (two-factor)` — pre/post activity only, with no modulator or error term, even
  when a teacher clamps the postsynaptic activity. If so, the signal is `supervised / imitation`.
- `perturbation` — gradient estimates from injected noise without a population, when not framed
  as reward-modulated plasticity.
- `evolutionary / black-box` — population-based or black-box search (GA, ES, CMA-ES, NEAT,
  Bayesian optimization).
- `ANN-to-SNN conversion`.
- `other` — anything else; describe it in `notes`. Examples: a fixed weight increment whenever an
  error neuron fires (whatever the vendor calls it), contrastive divergence, a classical
  adaptive-control law on neuron gains.

**`regime`** (single): how updates relate to the data stream.
- `offline` — fitted to a stored dataset, recordings or target patterns, usually over several
  epochs or batches, while the plant is not running. This holds even when an "online" or local rule runs on-chip over
  the stored data, and when babbling data is collected first and trained on afterwards. Examples: supervised datasets, recorded teacher or babbling
  data, conversion, a fixed dataset with feedback from a downstream network.
- `interleaved` — training alternates between collecting experience (episodes, rollouts, trials)
  and updates computed from it: after each batch, episode or trial, or from a replay buffer. This
  covers deep RL (PPO, SAC, TD3, DQN, even with an update every environment step from replay),
  evolution, BPTT through the agent's own rollouts (whatever the paper calls it), updates
  accumulated over a trial and applied at its end, and one-shot updates after a trial.
- `online` — parameters update continuously (per time step or event) from the current experience
  **while the plant runs**, with no separate dataset or batch phase. The network may be driving
  the plant, or observing a plant driven by babbling or a teacher. Examples: PES, FOLLOW, R-STDP
  during the run, online e-prop.

**`adapts_during_evaluation`** (bool): true if *learned* parameters change during the evaluation.
Designed fast synaptic dynamics (short-term facilitation, presynaptic inhibition) are not
adaptation.
- True for: online adaptation in the reported runs, test-time adaptation, and, for an `online`
  learner, results that are its learning curves.
- False when the parameters are frozen for the reported results. Training curves of
  `interleaved` learners (deep RL, evolution) don't make them adaptive: use false unless the paper
  says the policy keeps learning in the evaluation.
- If a rule could be applied per step or per trial, use what the paper's main experiments do.

**`evidence`** (string): one quote showing how this component is obtained or trained.

---

## 3. Analytic design

**`analytic_methods`** (multi-select; `[]` if no component is hand-designed, solved, or a fixed
reservoir). The labels say:
- the framework that computes the weights, if any: `NEF`, `spike coding network`;
- the origin of the computation: `control-theoretic` or `hand-wired circuit`;
- or that a fixed random network serves as a basis: `reservoir`;
- `other` for anything else analytic.

So PID in Nengo is [`NEF`, `control-theoretic`], an NEF-built CPG is [`NEF`,
`hand-wired circuit`], and PID with hand-wired neuron arrays or WTA circuits is
[`control-theoretic`]. `hand-wired circuit` is an origin label: use it only when the computation
itself is a bespoke design, not when a control law is implemented with hand-wired neurons.
- `control-theoretic` — the law comes from control theory: PID/PD, LQR, observers or Kalman
  filters, impulsive or bang-bang control, Lyapunov-based design.
- `NEF` — Neural Engineering Framework or Nengo encoders and decoders.
- `spike coding network` — efficient or balanced spike coding (Denève/Machens-style SCNs).
- `hand-wired circuit` — a bespoke designed circuit with set weights, spiking or not, that
  implements no control-theory law: CPGs, reflex arcs, winner-take-all circuits, filter models.
  Fixed relay or random wiring is not a hand-wired circuit.
- `reservoir` — a fixed random recurrent network used as a basis (LSM or ESN-style).
- `other` — e.g. a solved readout with no named framework (a Nengo solver counts as `NEF`), or a
  hand-written rule table.

---

## 4. Controller interface (spikes → actuation)

**`interface`** (single): how spikes become commands to the plant.
- `continuous` — spikes are turned into continuous-valued commands before they act on the plant:
  synaptic or low-pass filtering, firing-rate or spike-count windows, membrane-potential readout,
  population-vector or NEF decoding, a non-spiking readout layer, a value read off *which*
  neuron spiked (place codes, even one spike per step), or a burst detected over a window that
  sets a value. This also applies when the decoded value then selects a discrete action.
- `event-native` — individual spikes act directly as discrete actuation events: impulses, torque
  or thrust pulses (including a burst that gates one pulse), set-point increments, impulsive or
  bang-bang switching, or spikes driving a motor spike by spike (as PWM or stepper pulses).
- `mixed` — both kinds of channel, or a command that adds raw spike impulses to a filtered rate
  term (as in some spike-coding-network decoders).
- `not applicable` — no actuation (perception, no control task).
- `not reported` — the paper gives no basis at all. If the commands are continuous values (e.g.
  torques, velocities from a deep-RL actor) and no event-native actuation is described, use
  `continuous` even when the decoding isn't spelled out.

**`interface_evidence`** (string).

---

## 5. Hardware and deployment

**`platform`** (single): the most hardware-specific platform on which the paper runs its own
controller or SNN for reported results.
- `neuromorphic chip (digital)` — Loihi/Loihi 2, SpiNNaker/SpiNNaker 2, TrueNorth, Tianjic,
  Speck, Xylo and other digital neuromorphic ASICs.
- `neuromorphic chip (mixed-signal / analog)` — BrainScaleS, DYNAP-SE, ROLLS, memristive, PCM or
  other in-memory analog cores.
- `FPGA / digital accelerator` — an SNN on an FPGA, or on a custom digital accelerator that is
  not a neuromorphic chip product, including the authors' own neuromorphic FPGA designs.
- `neuromorphic emulator / SDK` — chip behaviour simulated in software (NengoLoihi emulator, Lava
  CPU backend), with no physical chip.
- `embedded CPU / microcontroller` — a CPU on the robot, a Raspberry Pi, a Jetson, an MCU.
- `CPU/GPU` — a workstation, server or "standard computer" CPU or GPU that the paper states,
  named or not ("on a GPU", "a desktop PC", "on one processing core").
- `software simulation (machine not stated)` — the network (or, for a non-SNN paper, the control
  law) is clearly simulated numerically
  (Brian2, Nengo, PyTorch, NEST, custom code, forward Euler…), but the paper never says on what
  machine.
- `not reported` — no indication of how the network was executed (e.g. pure theory).

**`platform_name`** (string): e.g. "Loihi (Kapoho Bay)", "SpiNNaker 48-node board",
"Xilinx Zynq FPGA", "RTX 3090", or "".

**`platform_coverage`** (single):
- `whole network` — the whole SNN runs on the chip, FPGA or emulator. Host-side input encoding,
  spike I/O conversion and output decoding are allowed.
- `partial` — part of the network computation runs off the platform: e.g. analog cores compute
  only the matrix products while the neuron dynamics run on a CPU, or some layers run on the host.
- `not applicable` — for `CPU/GPU`, `embedded CPU / microcontroller`,
  `software simulation (machine not stated)` and `not reported`.

When the SNN is split across platforms (e.g. part on a Raspberry Pi, part on a laptop), give the
most hardware-specific platform. Mention the split in `notes`.

**`hardware_in_loop`** (bool): true if the paper states that the network on a chip, FPGA, emulator
or embedded platform runs in closed loop with the plant during reported control runs: a real
robot, or a simulator exchanging data with the chip every control step. False for workstation
simulations, and for offline or benchmark-only runs on the hardware (e.g. control runs on a CPU
plus a separate chip benchmark). If the paper doesn't say whether the control runs used the chip,
use false.

**`platform_evidence`** (string).

---

## 6. Reported evaluation metrics

These go in the **`metrics`** object. Report what the paper does for its own system.
- **`tracking_error`**: `reported` (quantitative task performance: numbers in text or tables, or
  plots of an error or performance measure with numeric axes, i.e. tracking or regulation error,
  return, success rate, accuracy) / `not reported` (only qualitative statements, or trajectory
  plots without an error or performance measure).
- **`latency`**: `measured` (measured time per inference or control step, loop latency or real-time
  factor on the running system) / `estimated` (computed from a model, clock counts or
  layers × time-step) / `claimed` (a qualitative "low latency" statement about this system or its
  design) / `not reported`. A measured inference rate (frames or inferences per second) counts
  as measured. These don't count as measured latency: event throughput (events per second), a
  configured loop period, and generic remarks that SNNs have low latency.
- **`energy`**: `measured` (power or energy measured for this system) / `estimated` (computed from
  operation or spike counts, or a power model) / `claimed` (a statement about this system's energy
  with no own measurement or estimate, including quoting a chip's datasheet figures for the system
  used) / `not reported`. These are `not reported`: generic remarks that SNNs or chips are
  efficient (e.g. in the introduction), operation counts without an energy figure, and
  plant-side energy (cost of transport, motor energy).
- **`spike_activity`**: `reported` (quantitative spike rates, counts, sparsity or
  synaptic-operation numbers for the evaluated system, including burst rates, in text, tables
  or rate plots) / `not reported` (none, raster or tuning plots only, calibration-only numbers,
  or sparsity figures borrowed from other papers).
- **`stability`**: `formal` (the paper's own analytic proof or guarantee about stability,
  boundedness or convergence: of the closed loop, of the network dynamics, e.g. an SCN's
  bounded-error guarantee, or of the learning rule) / `empirical` (the paper explicitly examines
  whether the controlled system stays stable or bounded, without proof: a discussion of stable
  versus unstable regimes, oscillation or divergence, a stability trade-off, or statements that
  stability improved) / `not addressed`. Merely succeeding at a stabilization task is not a
  stability analysis.
  These are `not addressed`: theorems that only bound training updates, and guarantees that hold
  only for a reference controller the SNN doesn't implement exactly.
  Variance across training seeds is not stability. Settling time or steady-state error numbers
  alone are performance, not stability. A proof only cited from earlier work is
  `not addressed`.
- **`robustness`**: `tested` (explicit tests beyond the nominal conditions: added noise,
  disturbances, delays, parameter or plant changes, damage, neuron or synapse loss, varied
  hardware conditions, noise injected into the neurons) / `not tested`. These don't count:
  randomization that is part of training; the normal task variation (including target shifts,
  even if the paper calls it robustness); a plain comparison between hardware and software;
  sweeps over the controller's own hyperparameters (neuron count, time constants, surrogate
  shape); and attacks on the training process. Varying internal network noise and measuring
  performance does count.
- **`sim_to_real`**: `transferred` (the paper trains or designs the controller in simulation, then
  runs it on a real plant) / `real only` (the paper's controller is developed and evaluated on
  real hardware, including controllers fitted in software without a simulated plant) /
  `simulation only` / `not applicable` (no plant driven by the system: recorded data, perception
  only, no plant, or pure theory). If a paper has both simulation-only experiments and a real
  robot, use the real-robot case.
- **`metrics_evidence`** (string): a quote for the strongest hardware, latency or energy claim, or
  "".

---

## 7. Notes

**`notes`** (string): at most 2 sentences on anything that didn't fit the options, or where the
paper is ambiguous. Leave it empty otherwise.

---

## Worked edge cases (follow these)

- **Spiking actor + ANN critic, deep RL (TD3/SAC/PPO) in simulation, frozen at test.**
  - Actor: `controller`, spiking, deployed, `learned`, `reinforcement`, `backprop / BPTT`,
    `interleaved`, adapts=false.
  - Critic: `critic / value`, not spiking, not deployed, `learned`, `reinforcement`,
    `backprop / BPTT`, `interleaved`, adapts=false.
- **NEF arm controller with PES adaptation (REACH-like).**
  - Decoders: `readout / decoder`, `solved`.
  - Adaptive population: `controller`, `learned`, `supervised / imitation` (the error is the
    tracking error), `local error (LMS-like)`, `online`, adapts=true.
  - `analytic_methods`: `NEF`, plus `control-theoretic` if the paper uses a control law such as
    operational-space or PD control.
- **PID with hand-tuned gains on spiking neurons, on Loihi, flying a drone.**
  - One `hand-designed` component; `analytic_methods` = [`control-theoretic`]; `model-free`.
  - Signal, mechanism and regime are `not applicable`.
  - `hardware_in_loop` = true if the chip flies the drone in real time.
- **Evolutionary search of controller weights in simulation, then a real drone.** `searched`,
  `reinforcement` (fitness), `evolutionary / black-box`, `interleaved`; `sim_to_real` =
  `transferred`.
- **Reservoir with a FORCE/RLS-trained readout.**
  - Recurrent weights: `random / fixed`.
  - Readout trained online (FORCE/RLS while the system runs): `learned`,
    `supervised / imitation`, `local error (LMS-like)`, `online`.
  - Readout fitted once to recordings by regression: `solved`.
  - `analytic_methods`: [`reservoir`].
- **ANN trained, then converted.** Two components:
  - ANN: `learned`, not spiking, not deployed, its own signal, `backprop / BPTT`, `offline`.
  - SNN: `converted`, spiking, deployed, the same signal, `ANN-to-SNN conversion`, `offline`.
- **FOLLOW-style learning while babbling torques drive the arm.** `learned`,
  `supervised / imitation` (or `self-supervised / system identification` for a forward model),
  `local error (LMS-like)`, `online`. adapts=false if the reported test runs are frozen.
- **Policy trained by BPTT through a learned spiking world model on its own rollouts, with a
  replay buffer.**
  - World model: `self-supervised / system identification`, `backprop / BPTT`, `interleaved`.
  - Policy: `reinforcement`, `backprop / BPTT`, `interleaved`.
  - `plant_model_use` = `model learned`; spiking roles include `state estimation`.
- **Spike coding network controller that receives only the reference and computes the plant state
  internally (no plant feedback into the network).** `open-loop actuation`; analytic `solved`;
  `analytic_methods` = [`spike coding network`, `control-theoretic`].
- **Evolution minimising the integrated tracking error of closed-loop runs.** `searched`,
  `reinforcement`, `evolutionary / black-box`, `interleaved`.
- **Hebbian learning on a neuromorphic chip, switched on only while a teacher controller has
  settled.** `learned`, `supervised / imitation`, `eligibility + modulator (three-factor)` (the
  gate is a global modulator), `online`.
- **Learning-to-learn.**
  - Outer loop (e.g. BPTT across many tasks): `interleaved`, adapts=false.
  - Inner adaptation (one update after a trial, at test time): its own mechanism, `interleaved`,
    adapts=true.
- **Trained ANN downstream of the SNN in the deployed pipeline** (e.g. a CNN tracker fed by SNN
  event slices): list it as a deployed, non-spiking component.
- **SNN estimates ego-motion from an event camera; a separate (evolved) linear layer maps it to
  thrust commands; the drone flies.**
  - `spiking_roles` = [`sensing`, `state estimation`]; `control_level` = `closed-loop`.
  - The linear layer is a `controller` component: not spiking, `searched`.
- **Online RL whose results are learning curves, with no separate frozen test phase.** The
  evaluation *is* the learning, so adapts=true. The regime is `online` if the weights update every
  step, `interleaved` if updates are applied at the end of each trial. Deep RL that reports
  training curves but evaluates (or would deploy) a frozen policy keeps adapts=false.
