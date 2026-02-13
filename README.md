# 🧠 Recall-Extend Dynamics CoDA (RED-CoDA)

**Enhancing Small Language Models as Advanced Reasoning Agents through Context-Decoupled Hierarchical Architecture with Recall-Extend Dynamics.**

Built on [CoDA](https://github.com/xiao10ma/CoDA) (Context-Decoupled Agent) + RED (Recall-Extend Dynamics) frameworks, powered by Gemma-2-2B with GRPO reinforcement learning and LoRA efficient fine-tuning.

[![Model on HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Phonsiri/CoDA-Gemma2-RED-v3)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="assets/main_graph.jpg" width="720" alt="RED-CoDA Architecture"/>
</p>

---

## 📌 Problem & Motivation

Small Language Models (SLMs) struggle with complex reasoning tasks. Standard training approaches each have critical weaknesses:

| Approach | Problem |
|----------|---------|
| **SFT only** | Overfits to teacher patterns, poor generalization |
| **RL only** | Insufficient exploration space for small models |
| **SFT → RL (sequential)** | Catastrophic forgetting of learned patterns |

**RED-CoDA** solves this by **jointly training SFT + RL** with dynamic weighting controlled by two complementary mechanisms:

| RED Component | Mechanism |
|---|---|
| **Part 1: Dynamic Entropy Regulation** | Monitors entropy changes `δH_sft / δH_rl` to balance exploration (RL) vs. exploitation (SFT) — prevents mode collapse |
| **Part 2: Accuracy-Aware Policy Shift** | Adjusts via `G^(1 − 2·acc)` — low accuracy → boost SFT (learn from expert); high accuracy → boost RL (trust self-policy) |

<p align="center">
  <img src="assets/main_result.png" width="640" alt="Training Results"/>
</p>

---

## 🏗️ Architecture

RED-CoDA uses a **single shared LLM** (Gemma-2-2B) that operates in two decoupled contexts for advanced reasoning:

```
┌──────────────────────────────────────────────────────────────┐
│                  Gemma-2-2B  (Single Shared LLM + LoRA)      │
│                                                              │
│   🧠 Planner (Strategic)          ⚡ Executor (Ephemeral)    │
│   ─────────────────────           ──────────────────────     │
│   • Decomposes complex            • Receives subtask via     │
│     questions into subtasks         <task>...</task>          │
│   • Maintains full context        • Reasons independently   │
│   • Delegates via <task>          • Returns <answer>         │
│   • Synthesizes final answer      • Context: decoupled       │
│   • Context: persistent             (reset each task)        │
│     across turns                                             │
└──────────────────────────────────────────────────────────────┘
```

**Key Insight (Context Decoupling):** The Planner keeps full conversation history across turns, while each Executor instance starts fresh with *only* the subtask — preventing context pollution and enabling clean, focused reasoning. Both roles are served by the **same model** with different system prompts.

### Actor-Critic Two-Role Reasoning

RED-CoDA also supports a **Max Procedural Thinking** mode where the model alternates between Actor (Generator) and Critic (Evaluator) personas within a single generation:

1. **Actor — Initial Reasoning:** Step-by-step problem decomposition
2. **Critic — Evaluation:** Strict logic verification, skeptical checking
3. **Actor — Refined Reasoning:** Incorporates feedback into final answer

---

## 🔄 RED Training Loop

Each training step combines RL exploration with SFT-guided learning, dynamically weighted:

```
┌─────────────────────────────────────────────────────────────┐
│  1 Training Step                                             │
│                                                              │
│  🎲 RL Rollout (vLLM) → Generate multi-turn responses       │
│  📊 Compute Reward:                                         │
│       • Answer Quality (F1/EM/Qwen-Judge)                   │
│       • Format Compliance (XML tag scoring)                  │
│  📈 GRPO Advantage (group normalization by prompt)           │
│                                                              │
│  ┌─── RED Weight Computation ─────────────────────────┐     │
│  │  Part 1: w_entropy = clamp(δH_sft / δH_rl, 1, G)  │     │
│  │  Part 2: w_accuracy = G^(1 − 2 · batch_accuracy)   │     │
│  │  final_w = w_entropy × w_accuracy                   │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  🧠 Actor Update (single optimizer step):                    │
│     L = L_policy(GRPO)                                       │
│       − β · H(π)              ← entropy bonus                │
│       + final_w · α · L_SFT   ← RED-weighted cross-entropy  │
└─────────────────────────────────────────────────────────────┘
```

### Composite Reward Function

| Component | Formula | Description |
|-----------|---------|-------------|
| **Answer Quality** | `6 × score − 3` | F1, EM, CEM, or Qwen-Judge score (dominant signal) |
| **Format Compliance** | `0.1 × score` | Graduated XML tag scoring (`<think>` + `<answer>` = 0.5 each) |

### Reward Styles

| Style | Scorer | Description |
|-------|--------|-------------|
| `EM` | Exact Match | Binary correctness check |
| `F1` | Token-level F1 | Partial credit for overlapping tokens |
| `CEM` | Cover EM | Any ground truth answer matched |
| `qwen_judge` | Qwen-7B-Instruct (vLLM) | LLM-as-judge scoring with strict auditing rubric (0–10, normalized) |

---

## ⚙️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Base Model** | `google/gemma-2-2b` |
| **Judge Model** | `Qwen/Qwen2.5-7B-Instruct` (served via vLLM) |
| **RL Algorithm** | GRPO (Group Relative Policy Optimization) |
| **Fine-tuning** | LoRA (via `peft`) — rank 16 by default |
| **Co-training** | SFT cross-entropy with RED dynamic weighting |
| **Distributed** | Ray + FSDP (Fully Sharded Data Parallel) |
| **Config** | Hydra + OmegaConf |
| **Logging** | Weights & Biases |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- CUDA 12.x compatible GPU (H100 recommended)

### 1. Clone & Install

```bash
git clone https://github.com/Phonsiriwillbejommarn/Recall-Extend-Dynamics-CoDA.git
cd Recall-Extend-Dynamics-CoDA
pip install -e .
```

### 2. Login Services

```bash
wandb login            # For training dashboard
huggingface-cli login  # For checkpoint push
```

### 3. Prepare Data

```bash
# Process training data
bash preprocess/scripts/data_process.sh

# Generate SFT training data (pure reasoning patterns)
python cmd/generate_sft_data.py
```

### 4. Start Training

```bash
bash cmd/train.sh
```

### 5. Evaluate

```bash
bash cmd/eval.sh
```

### 6. Inference Demo

```python
python inference_demo.py
# or for direct reasoning:
python infer.py
```

---

### 7. Cloud / Continuous Training Workflow

If your dataset updates frequently (e.g., new data arriving every week), follow this sequence:

1.  **Ingest New Data:**
    Update your raw data source or the `preprocess/scripts/data_process.sh` script to point to new files.
    ```bash
    # 1. Convert raw data (JSON/CSV) to Parquet format
    bash preprocess/scripts/data_process.sh
    ```

2.  **Regenerate SFT Traces:**
    Gemma needs to see *how* to reason over the new data. This step generates the `sft_data.parquet` with pure reasoning traces (`<think>...</think>`).
    ```bash
    # 2. Generate pure reasoning traces for SFT co-training
    python cmd/generate_sft_data.py
    ```

3.  **Start/Resume Training:**
    ```bash
    # 3. Resume training (will load latest checkpoint automatically)
    bash cmd/train.sh
    ```

---

## ⚙️ Configuration

All configs are set in [`cmd/train.sh`](cmd/train.sh) and passed via Hydra:

### Core Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_batch_size` | 32 | Prompts per training step |
| `n_agent` | 2 | Responses per prompt (GRPO group size) |
| `max_turns` | 2 | Planner reasoning rounds |
| `total_training_steps` | 480 | Total training steps |
| `learning_rate` | 1e-6 | Actor learning rate |

### RED Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sft.enabled` | `true` | Enable SFT co-training |
| `sft.loss_coef` | `0.1` | Base SFT loss coefficient (α) |
| `red.G` | `5.0` | Upper bound for RED weight clamping |
| `red.sft_entropy_ema_decay` | `0.99` | SFT entropy EMA smoothing |
| `red.rl_entropy_ema_decay` | `0.99` | RL entropy EMA smoothing |
| `algorithm.accuracy_aware_policy_shift` | `true` | Enable RED Part 2 |

### LoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.rank` | `16` | LoRA rank |
| `lora.target_modules` | `q_proj,v_proj` | Modules to apply LoRA |
| `lora.lora_alpha` | `32` | LoRA scaling factor |

### Ablation Configurations

```bash
# Baseline: GRPO only (no SFT co-training)
sft.enabled=false

# GRPO + Fixed SFT (no dynamic RED weighting)
sft.enabled=true red.G=1.0

# GRPO + RED Full (recommended)
sft.enabled=true red.G=5.0 algorithm.accuracy_aware_policy_shift=true
```

---

## 📁 Project Structure

```
Recall-Extend-Dynamics-CoDA/
├── cmd/
│   ├── train.sh                    # Main training script & config
│   ├── eval.sh                     # Evaluation script
│   └── generate_sft_data.py        # Generate SFT co-training data (pure reasoning)
├── search_r1/
│   └── llm_agent/
│       ├── generation.py           # Hierarchical Planner/Executor agent loop
│       └── tensor_helper.py        # Tensor utilities for generation
├── verl/                            # Core framework
│   ├── trainer/
│   │   ├── main_ppo.py             # Entry point + RewardManager
│   │   └── ppo/
│   │       ├── ray_trainer.py      # Distributed training loop + RED integration
│   │       └── core_algos.py       # GRPO advantages + RED weight algorithms
│   ├── workers/
│   │   ├── actor/
│   │   │   └── dp_actor.py         # Actor update: RL loss + RED-weighted SFT loss
│   │   └── fsdp_workers.py         # FSDP workers with LoRA integration
│   └── utils/
│       └── reward_score/
│           ├── qa_em.py            # F1, EM, format scoring functions
│           └── qwen_judge.py       # LLM-as-judge via Qwen/vLLM
├── dataset/
│   └── FORMAT_README.md            # Required response format specification
├── preprocess/                      # Data processing scripts
├── assets/                          # Architecture diagrams & result plots
├── test_lora_load.py               # LoRA integration unit test
├── infer.py                        # Pure reasoning inference
├── inference_demo.py               # Actor-Critic reasoning demo
├── requirements.txt
└── pyproject.toml
```

---

## 📊 W&B Metrics

### Key Metrics to Monitor

| Metric | Description |
|--------|-------------|
| `critic/rewards/mean` | Overall reward per step |
| `answer_f1/mean` | Answer quality (F1 score) |
| `format_scores/mean` | XML format compliance |
| `red/entropy_weight` | RED Part 1 — entropy-based weight |
| `red/accuracy_factor` | RED Part 2 — accuracy-based multiplier |
| `red/final_weight` | Combined RED weight (w_entropy × w_accuracy) |
| `red/batch_accuracy` | Fraction of correctly answered samples |
| `actor/sft_loss` | SFT cross-entropy loss |
| `actor/sft_entropy` | SFT prediction entropy |
| `actor/rl_entropy` | RL rollout entropy |

---

## 📄 Data Format

Training data follows a strict XML-structured response format (see [`dataset/FORMAT_README.md`](dataset/FORMAT_README.md)):

```xml
<think>
[Actor - Initial Reasoning]
Step 1: I need to analyze the problem...
Step 2: Breaking it down...
Initial Answer: 50

[Critic - Evaluation]
Step 1 Feedback: Incorrect order of operations (PEMDAS).
Overall Assessment: Fail, recalculation needed.

[Actor - Refined Reasoning]
Step 1 (Revised): Calculate 5 * 2 first = 10
Step 2 (Revised): Add 20 + 10 = 30
Final Answer: 30
</think>
<answer>30</answer>
```

---

## 🔧 Restart After Server Reboot

```bash
cd Recall-Extend-Dynamics-CoDA
git pull origin main
bash preprocess/scripts/data_process.sh    # Recreate parquet files
python cmd/generate_sft_data.py            # Recreate SFT data
bash cmd/train.sh                          # Auto-resumes from HF Hub checkpoint
```

---

## 📝 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- **[CoDA](https://github.com/xiao10ma/CoDA)** — Context-Decoupled Hierarchical Agent architecture
- **[Search-R1](https://github.com/PeterGriffinJin/Search-R1)** — Base framework for agent training
- **[verl](https://github.com/volcengine/verl)** — Distributed RL training framework
- **[Google Gemma-2-2B](https://huggingface.co/google/gemma-2-2b)** — Base language model
- **[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)** — Judge model for reward scoring
- **[LoRA / PEFT](https://github.com/huggingface/peft)** — Efficient fine-tuning