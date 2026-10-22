# RL post-training: GRPO on MBPP

A separate experiment from the V0–V3 agents in the rest of this repository. It
asks a different question: **can reinforcement learning against unit tests make
a small code model better at writing code that passes them?**

Nothing here changes the V0–V3 code or results.

> **Status:** sandbox, reward function, training and evaluation code are done
> and tested. The reward signal has been checked on the base model. The full
> runs (base, SFT, GRPO, SFT→GRPO on the test split) have **not run yet**;
> the results section below is empty until they do.

---

## The setup in one paragraph

Each MBPP problem is a short task ("write a function to …") with three assert
statements. The model writes a function; the function runs inside a sandbox;
the **reward is the fraction of asserts that pass**. GRPO samples 8 answers per
problem, and pushes the model towards the answers that scored above that
group's average. A model only learns from a problem when its 8 answers score
differently, so the first thing checked was whether that happens often enough.

| | |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct`, revision `ea3f2471`, LoRA r=16 on all linear layers |
| Data | MBPP "full" config, revision `4bb6404f`, official splits: train 374 / validation 90 / test 500 |
| Training data | train split only |
| Model selection | validation split only (which epoch to keep) |
| Test split | evaluated once per final model, after selection |
| Method | TRL 1.16 `GRPOTrainer`, `loss_type="grpo"`, KL β=0.04, 8 generations per prompt, T=0.7 |
| Metric | pass@1 = greedy answer passes **all** asserts |

### Scores are not comparable to published MBPP numbers

The prompt shows **only the first** of the three asserts (so the model knows
the function name and signature). Published MBPP results show all three. This
was chosen on purpose: with all three visible, a model can pass by hard-coding
the three expected outputs, and RL is exactly the kind of optimiser that finds
that. With one visible, hard-coding the visible case still fails the two hidden
ones. The cost is that the numbers here are lower than, and not comparable
with, leaderboard MBPP numbers.

---

## The sandbox

Model-generated code never runs directly on the host. Every assert runs in its
own fresh Python process inside a container, so one test cannot see what
another left behind. A test counts as passed only if the child process prints
a random per-batch token **after** the assert ran, so `sys.exit(0)` or
`os._exit(0)` before the assert scores 0, not 1.

Two backends with the same harness (`sandbox/harness.py`):

| Protection | Docker (local) | Apptainer (cluster, no root) |
|---|---|---|
| No network | `--network none` | `--net --network none` |
| Read-only root filesystem | `--read-only` | image mounted read-only |
| Nothing from the host mounted | no volumes; batches go over stdin | `--containall --no-home --no-mount home,cwd,bind-paths,hostfs`, no `--nv` |
| No capabilities, no-new-privileges | `--cap-drop ALL`, `no-new-privileges` | unprivileged by construction |
| Own PID namespace, orphans reaped | `--init` | `--containall` (fresh container per batch) |
| seccomp | per-container Docker default profile **and** the harness filter | **harness filter only** |
| Memory / CPU / process limits | `--memory 2g --cpus 4 --pids-limit 256` **and** per-process rlimits | **per-process rlimits only**, inside SLURM's job cgroup |
| Separate user id | uid 10001, owns nothing | **runs as the cluster user's uid** |

Per-process limits in every child: 512 MB address space, CPU time, 1 MB output
file size, 64 processes, no core dumps, wall-clock timeout (5 s).

**The harness seccomp filter.** Each child installs Docker's own default
profile (moby v28.1.1, `sandbox/seccomp-default.json`) on itself, through
libseccomp, before any model code runs. This exists because of two findings:

1. The Docker Desktop daemon on the development machine runs containers with
   **seccomp unconfined by default** (`docker info`: "daemon is not using the
   default seccomp profile"). A container started without an explicit profile
   showed `Seccomp: 0`. So the profile is passed per container, and checked.
2. Unprivileged Apptainer **cannot apply a seccomp profile at all**
   (`--security` is root-only), and it adds none of its own: a container
   started with the same flags showed `Seccomp: 0`, and inside it
   `unshare(CLONE_NEWUSER)` succeeded and an `AF_VSOCK` socket opened. With
   the harness filter, both are refused, and the tests check this.

**Tests.** `tests/test_rewards.py`, 33 tests: reward values (correct, partial,
syntax error, no code block, setup code), containment (infinite loop, memory
blowup, fork bomb, huge output, network, read-only filesystem, no
capabilities, host files invisible, no GPU device, seccomp active and blocking
`unshare` / `AF_VSOCK` / `mount`), reward hacking (`sys.exit`, `os._exit`,
`exit`, `quit`, printing a fake success marker, state leaking between tests),
and batching. Every cluster job runs the full suite on its GPU node first and
stops if anything fails or is skipped.

The fork bomb test found a real bug: killed fork-bomb children were reparented
to the container's PID 1 (`sleep`), which never reaped them; 252 zombies sat
against a 256 process limit. Fixed with `--init`.

---

## Measured so far

### Is there a reward signal? (base model, 50 train problems)

Greedy answer plus 8 sampled answers per problem, scored in the sandbox.
Train split only. Raw data: `results/reward_signal_base*.json`.

| | T=0.8 | T=0.7 |
|---|---|---|
| Greedy pass@1 | 0.40 | 0.40 |
| Sampled pass@1 | 0.2975 | 0.345 |
| Problems whose 8 samples got different rewards | **0.66** | **0.66** |
| Problems where all 8 scored 0 | 0.28 | 0.28 |
| Problems where all 8 scored 1 | 0.06 | 0.06 |
| Problems with at least one fully passing sample | 0.54 | 0.66 |

Two thirds of problems give GRPO something to learn from, so the model and
task set were kept. Training uses T=0.7.

### Speed on a MacBook (M4, 16 GB, MPS, fp32)

5 GRPO steps of 64 completions each: 147.7 s per step on average (generation
42.0 s, sandbox scoring 2.7 s, forward/backward 103.1 s). Micro-batch 8 ran
out of memory; micro-batch 2 with gradient checkpointing fit. This is why the
full run moved to a cluster GPU.

---

## Results

*Pending: filled in from `results/*_test.json` once the full runs finish.*

---

## Limitations

- **Not comparable to published MBPP scores** (one visible assert; see above).
- **The Apptainer sandbox is weaker than the Docker one.** Without root there is
  no container-level seccomp, no cgroup limits of its own, and code runs as the
  cluster user's uid. The harness filter and rlimits replace the first two; SLURM's
  job cgroup bounds the whole job. Nothing from the host is mounted, and a test
  checks that host files are not visible, but a sandbox escape would act as that
  user.
- **The Apptainer backend was tested locally on Apptainer 1.5.4 with a
  directory image**, because the PPA has no 1.4.5 and packed SIF images could not
  execute under Docker Desktop's FUSE. The cluster (1.4.5, SIF) reruns the same
  tests before every job.
- **An object whose `__eq__` always returns `True` passes equality asserts.** The
  sandbox cannot detect this; a test documents it, and the reward-hacking audit
  looks for it in the code.
- **One seed, no hyperparameter search.** The learning rate (1e-5 GRPO, 1e-4 SFT)
  was chosen, not tuned. Only the epoch is selected, on validation.
- **Not bit-for-bit reproducible.** Seeds are fixed, but CUDA kernels are not
  forced to be deterministic.
- **The base model often pastes the visible assert into its own answer.** If that
  assert fails, the code raises on import and every test scores 0. This is fair
  but makes rewards more all-or-nothing.

---

## How to run

Local (Docker): `pip install -r requirements.txt` into a Python 3.11 env, start
Docker, then `python -m pytest -q` from this directory.

Cluster (SLURM + Apptainer): one-time setup runs as CPU jobs, because the login
node's limits kill the conda solver: `sbatch slurm/setup_cpu.sbatch envpack`,
then `sandbox`, then `download`. `envpack` builds the env on the compute node's
local disk and writes it to scratch as a single tar (conda-pack); each GPU job
unpacks it to its own local disk. Installing straight onto the shared scratch
filesystem (`setup_cpu.sbatch env`) also works but is very slow, because it
writes tens of thousands of small files there. Then `sbatch slurm/smoke.sbatch`.
Full runs: `eval_base.sbatch`, `sft.sbatch`, `grpo.sbatch`, then
`sft_grpo.sbatch` (needs the SFT result). Every job can be resubmitted after a
timeout; training resumes from its newest checkpoint.

| File | What it does |
|---|---|
| `sandbox.py` | Docker and Apptainer backends |
| `sandbox/harness.py`, `sandbox/seccomp_filter.py` | runs inside the container |
| `rewards.py` | code extraction and the reward |
| `data.py`, `models.py` | pinned data, prompt, model loading |
| `check_reward_signal.py` | the reward-signal check above |
| `train_grpo.py`, `train_sft.py` | training, resumable |
| `select_and_eval.py` | choose epoch on validation, then evaluate on test |
| `evaluate.py` | greedy pass@1 |
| `configs/` | every run's settings |
| `slurm/` | cluster setup and job scripts |
