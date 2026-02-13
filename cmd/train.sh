#!/bin/bash
# =============================================================
# CoDA Training Script (Gemma-2-2b + RED Logic) สำหรับ H100
# =============================================================

# 1. การตั้งค่า Environment สำหรับ H100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="0"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1

# 2. ข้อมูลโมเดลและไดเรกทอรี
num_gpus=1
export DATA_DIR=data
export BASE_MODEL='google/gemma-2-2b'
export EXPERIMENT_NAME="Recall-Extend-Dynamics-CoDA"

# 3. API Keys (Paste your keys below)
# ท่านสามารถแก้ค่า "api" ด้านล่างให้เป็น Key ของท่านได้เลย
MANUAL_WANDB_KEY="api"
MANUAL_HF_TOKEN="api"

# Logic: Use Manual Key if set, otherwise use Environment Variable, otherwise fallback to "API"
if [ "$MANUAL_WANDB_KEY" != "api" ] && [ "$MANUAL_WANDB_KEY" != "API" ]; then
    MY_WANDB_KEY="$MANUAL_WANDB_KEY"
else
    MY_WANDB_KEY="${WANDB_API_KEY:-API}"
fi

if [ "$MANUAL_HF_TOKEN" != "api" ] && [ "$MANUAL_HF_TOKEN" != "API" ]; then
    MY_HF_TOKEN="$MANUAL_HF_TOKEN"
else
    MY_HF_TOKEN="${HF_TOKEN:-API}"
fi

# 4. Login W&B
export WANDB_API_KEY=$MY_WANDB_KEY
export WANDB_PROJECT="CoDA_RED_Project"
export WANDB_MODE="online" # เปิด W&B เพื่อดูกราฟผ่านเว็บ

python3 -c "
try:
    key = '${MY_WANDB_KEY}'
    if key and key != 'API':
        import wandb
        wandb.login(key=key)
        print('✅ W&B logged in successfully!')
    else:
        print('⚠️ W&B Key not found in env. Setting WANDB_MODE=offline.')
        import os
        os.environ['WANDB_MODE'] = 'offline'
except Exception as e:
    print(f'❌ Failed to login to W&B: {e}')
"

# 5. Login Hugging Face
export HF_TOKEN=$MY_HF_TOKEN

python3 -c "
try:
    token = '${MY_HF_TOKEN}'
    if token and token != 'API':
        from huggingface_hub import login
        login(token=token)
        print('✅ Hugging Face logged in successfully!')
    else:
        print('⚠️ HF Token not found in env. Using public models only.')
except Exception as e:
    print(f'❌ Failed to login to Hugging Face: {e}')
"

# สร้างโฟลเดอร์สำหรับเก็บ Log
mkdir -p log/

# 7. Start Qwen-7B-Instruct Judge Server (vLLM)
echo "🚀 Starting Qwen-7B-Instruct Judge Server..."
export VLLM_PORT=8000
# GPU Memory Split: Qwen (Judge) 40%, Gemma (Actor) 50% = 90% Total (Safe for 1x H100)
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port $VLLM_PORT \
    --gpu-memory-utilization 0.4 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --trust-remote-code \
    --guided-decoding-backend lm-format-enforcer \
    --disable-log-stats &

VLLM_PID=$!
echo "Waiting for vLLM to start (PID: $VLLM_PID)..."
sleep 45 # Give it time to load weights

# 8. Check for Resume Checkpoint (HF Hub)
HF_REPO="Phonsiri/$EXPERIMENT_NAME"
RESUME_STEP=0
MODEL_PATH=$BASE_MODEL

if [ "$MY_HF_TOKEN" != "API" ]; then
    echo "🔍 Checking for existing checkpoints on HF Hub: $HF_REPO..."
    # Ensure auto_resume.py handles empty repo correctly
    RESUME_OUTPUT=$(python3 cmd/auto_resume.py --repo $HF_REPO 2>&1)
    RESUME_EXIT=$?

    if [ $RESUME_EXIT -eq 0 ]; then
        RESUME_PATH=$(echo "$RESUME_OUTPUT" | grep "RESUME_PATH=" | cut -d'=' -f2)
        FETCHED_STEP=$(echo "$RESUME_OUTPUT" | grep "RESUME_STEP=" | cut -d'=' -f2)
        if [ -n "$RESUME_PATH" ] && [ -d "$RESUME_PATH" ]; then
            MODEL_PATH=$RESUME_PATH
            RESUME_STEP=$FETCHED_STEP
            echo "✅ Resuming from checkpoint: step $RESUME_STEP"
            echo "   Model path: $MODEL_PATH"
        else
            echo "ℹ️ No valid checkpoint found on disk, training from base model."
        fi
    else
        echo "ℹ️ New experiment or repo empty, training from scratch."
    fi
else
    echo "⚠️ HF Token missing, skipping auto-resume check."
fi

# 9. Main Training Command
# Convert Data to Parquet first
if [ ! -f "data/train.parquet" ]; then
    echo "⚠️ data/train.parquet not found. Running pre-processing..."
    export PYTHONPATH=$PYTHONPATH:.
    bash preprocess/scripts/data_process.sh
fi

echo "🔄 Generating SFT Data (Typhoon + Synthetic)..."
python3 cmd/generate_sft_data.py

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    reward_model.reward_style="qwen_judge" \
    data.train_files=data/sft_train.parquet \
    data.val_files=data/sft_train.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=3072 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=3072 \
    max_turns=2 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    algorithm.red_enabled=true \
    algorithm.entropy_weight_regulation=true \
    algorithm.accuracy_aware_policy_shift=true \
    trainer.total_training_steps=480 \
    actor_rollout_ref.actor.refine_score=0.1 \
    actor_rollout_ref.actor.format_score=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.9 \
    ++actor_rollout_ref.rollout.top_p=0.9 \
    ++actor_rollout_ref.rollout.top_k=50 \
    ++actor_rollout_ref.rollout.repetition_penalty=1.2 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.resume_step=$RESUME_STEP \
    trainer.val_before_train=false \
    trainer.test_freq=50 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    sft.enabled=true \
    sft.train_files=data/sft_train.parquet \
    sft.loss_coef=0.1 \
    sft.micro_batch_size=4 \
    sft.max_length=4096 \
    red.G=5.0 \
    red.sft_entropy_ema_decay=0.99 \
    red.rl_entropy_ema_decay=0.99 \
    +actor_rollout_ref.actor.enable_max_procedural=true \
    2>&1 | tee log/$EXPERIMENT_NAME.log

# Cleanup
kill $VLLM_PID