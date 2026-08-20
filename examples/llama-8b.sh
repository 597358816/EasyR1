set -x
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_MODE=online
export WANDB_API_KEY="b80b9192efe12f9fc47ef0fc711bde76686fb981"
export TMPDIR=/vepfs-mlp2/c20250203/250602012/tmp
export PIP_CACHE_DIR=/vepfs-mlp2/c20250203/250602012/cache


MODEL_PATH=/vepfs-mlp2/c20250203/250602012/models/meta-llama/Llama-3.1-8B-Instruct
NAME="Llama-3.1-8B-DAPO2"

CHECKPOINT_ROOT="/vepfs-mlp2/c20250203/250602012/checkpoints/Llama-3.1-8B-Instruct/${NAME}"
LATEST_STEP_FILE="${CHECKPOINT_ROOT}/latest_global_step.txt"

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

# 默认不设置续训参数
LOAD_CHECKPOINT_ARGS=()

# 如果 latest_global_step.txt 存在，则自动读取最近 checkpoint
if [[ -f "${LATEST_STEP_FILE}" ]]; then
    # 去除换行符、回车符及首尾空格
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

        echo "检测到续训文件：${LATEST_STEP_FILE}"
        echo "将从 checkpoint 续训：${LOAD_CHECKPOINT_PATH}"
    else
        echo "${LATEST_STEP_FILE} 内容为空，不设置 load_checkpoint_path"
    fi
else
    echo "未检测到 ${LATEST_STEP_FILE}，从头训练"
fi

/vepfs-mlp2/c20250203/250602012/Anaconda/envs/easyr1/bin/python -m verl.trainer.main \
    config=/vepfs-mlp2/c20250203/250602012/EasyR1/examples/config.yaml \
    data.train_files=/vepfs-mlp2/c20250203/250602012/data/math/data/train-00000-of-00001.parquet \
    data.val_files=/vepfs-mlp2/c20250203/250602012/data/math/data/test-00000-of-00001.parquet \
    data.seed=2 \
    data.max_response_length=2048 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.project_name="new-AEPO" \
    trainer.save_freq=20 \
    trainer.save_limit=4 \
    trainer.val_freq=-1 \
    trainer.experiment_name="${NAME}" \
    trainer.save_checkpoint_path="/vepfs-mlp2/c20250203/250602012/checkpoints/Llama-3.1-8B-Instruct/${NAME}" \
    trainer.total_episodes=4 \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    trainer.algorithm="DAPO" \
    algorithm.disable_kl=true \
    "${LOAD_CHECKPOINT_ARGS[@]}"
