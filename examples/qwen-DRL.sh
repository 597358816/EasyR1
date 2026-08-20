set -x
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_MODE=online
export WANDB_API_KEY="b80b9192efe12f9fc47ef0fc711bde76686fb981"
export TMPDIR=/vepfs-mlp2/c20250203/250602012/tmp
export PIP_CACHE_DIR=/vepfs-mlp2/c20250203/250602012/cache


MODEL_PATH=/vepfs-mlp2/c20250203/250602012/models/Qwen/Qwen2.5-3B
MODEL_NAME=Qwen2.5-3B

NAME="qwen2.5-3b-GRPO-ref"

CHECKPOINT_ROOT="/vepfs-mlp2/c20250203/250602012/checkpoints/OPD/Qwen2.5-3B/${NAME}"
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
    worker.actor.model.model_path="${MODEL_PATH}" \
    data.train_files=/vepfs-mlp2/c20250203/250602012/data/math/data/train-00000-of-00001.parquet \
    data.val_files=/vepfs-mlp2/c20250203/250602012/data/math/data/test-00000-of-00001.parquet \
    data.max_response_length=4096 \
    data.rollout_batch_size=128 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.rollout.n=8 \
    worker.rollout.max_num_batched_tokens=8192 \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name="${NAME}" \
    trainer.project_name="DRL" \
    trainer.val_freq=-1 \
    trainer.save_limit=8 \
    trainer.save_freq=20 \
    trainer.total_episodes=3 \
    trainer.val_before_train=false \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.global_batch_size=64 \
    trainer.save_checkpoint_path="${CHECKPOINT_ROOT}" \
    worker.teacher.use_teacher=false \
    worker.teacher.model.model_path=/vepfs-mlp2/c20250203/250602012/models/Qwen/Qwen2.5-7B-Instruct \
    trainer.algorithm="GRPO" \
    trainer.DRL_rho_sample="positive" \
    trainer.DRL_rho_clip=3 \
    trainer.DRL_rho_normalized=true \
    algorithm.disable_kl=false \
    "${LOAD_CHECKPOINT_ARGS[@]}"


for step in $(seq 20 20 160)
do
    echo ${step}
    python3 ../scripts/model_merger.py --local_dir "/vepfs-mlp2/c20250203/250602012/checkpoints/OPD/${MODEL_NAME}/${NAME}/global_step_${step}/actor/"
done