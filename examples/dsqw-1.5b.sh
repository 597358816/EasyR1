set -x
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_MODE=online
export WANDB_API_KEY="b80b9192efe12f9fc47ef0fc711bde76686fb981"
export TMPDIR=/vepfs-mlp2/c20250203/250602012/tmp
export PIP_CACHE_DIR=/vepfs-mlp2/c20250203/250602012/cache


MODEL_PATH=/vepfs-mlp2/c20250203/250602012/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
NAME="DSQW1_5b-DSQW14B-DRL"

CHECKPOINT_ROOT="/vepfs-mlp2/c20250203/250602012/checkpoints/OPD/DSQW-1_5b/${NAME}"
LATEST_STEP_FILE="${CHECKPOINT_ROOT}/latest_global_step.txt"

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

# 默认不设置续训参数
LOAD_CHECKPOINT_ARGS=()

if [[ -f "${LATEST_STEP_FILE}" ]]; then
    LATEST_GLOBAL_STEP="$(tr -d '\r\n[:space:]' < "${LATEST_STEP_FILE}")"
    if [[ -n "${LATEST_GLOBAL_STEP}" ]]; then
        if [[ "${LATEST_GLOBAL_STEP}" == /* ]]; then
            LOAD_CHECKPOINT_PATH="${LATEST_GLOBAL_STEP}"
        elif [[ "${LATEST_GLOBAL_STEP}" == global_step_* ]]; then
            LOAD_CHECKPOINT_PATH="${CHECKPOINT_ROOT}/${LATEST_GLOBAL_STEP}"
        else
            LOAD_CHECKPOINT_PATH="${CHECKPOINT_ROOT}/global_step_${LATEST_GLOBAL_STEP}"
        fi

        LOAD_CHECKPOINT_ARGS+=(
            "trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH}"
        )

        echo "load checkpoint：${LATEST_STEP_FILE}"
    else
        echo "no load_checkpoint_path"
    fi
else
    echo "no ckpt in ${LATEST_STEP_FILE}"
fi

/vepfs-mlp2/c20250203/250602012/Anaconda/envs/easyr1/bin/python -m verl.trainer.main \
    config=/vepfs-mlp2/c20250203/250602012/EasyR1/examples/config.yaml \
    worker.actor.model.model_path="${MODEL_PATH}" \
    data.train_files=/vepfs-mlp2/c20250203/250602012/data/train_dapo.parquet \
    data.val_files=/vepfs-mlp2/c20250203/250602012/data/math/data/test-00000-of-00001.parquet \
    data.max_response_length=8192 \
    data.rollout_batch_size=128 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.rollout.n=8 \
    worker.rollout.max_num_batched_tokens=10240 \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name="${NAME}" \
    trainer.project_name="DRL" \
    trainer.val_freq=-1 \
    trainer.save_limit=8 \
    trainer.save_freq=20 \
    trainer.total_episodes=2 \
    trainer.val_before_train=false \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.global_batch_size=64 \
    trainer.save_checkpoint_path="${CHECKPOINT_ROOT}" \
    worker.teacher.use_teacher=true \
    worker.teacher.model.model_path=/vepfs-mlp2/c20250203/250602012/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    worker.teacher.model.tokenizer_path=/vepfs-mlp2/c20250203/250602012/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    trainer.algorithm="DRL" \
    trainer.DRL_rho_sample="positive" \
    trainer.DRL_rho_clip=3 \
    trainer.DRL_rho_normalized=true \
    "${LOAD_CHECKPOINT_ARGS[@]}"


    # worker.teacher.model.model_path=/vepfs-mlp2/c20250203/250602012/checkpoints/Qwen3-8B/qwen3-8b-GRPO/global_step_139/actor/huggingface \
    # worker.teacher.model.tokenizer_path=/vepfs-mlp2/c20250203/250602012/checkpoints/Qwen3-8B/qwen3-8b-GRPO/global_step_139/actor/huggingface \
    # trainer.DRL_rho_sample="all" \
    # trainer.DRL_rho_clip=1.3 \
# 可选参数：
# worker.reward.length_reward="LP1"
# worker.actor.use_entropy_loss=true

