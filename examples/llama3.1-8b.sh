set -x

export WANDB_BASE_URL=https://api.bandw.top
export WANDB_MODE=online
export WANDB_API_KEY="b80b9192efe12f9fc47ef0fc711bde76686fb981"
export TMPDIR=/vepfs-mlp2/c20250203/250602012/tmp
export PIP_CACHE_DIR=/vepfs-mlp2/c20250203/250602012/cache


MODEL_PATH=/vepfs-mlp2/c20250203/250602012/models/meta-llama/Llama-3.1-8B-Instruct  # replace it with your local file path
NAME="Llama3.1-8b-AEPO"
FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

/vepfs-mlp2/c20250203/250602012/Anaconda/envs/easyr1/bin/python -m verl.trainer.main \
    config=/vepfs-mlp2/c20250203/250602012/EasyR1/examples/config.yaml \
    data.train_files=/vepfs-mlp2/c20250203/250602012/data/math/data/train-00000-of-00001.parquet \
    data.val_files=/vepfs-mlp2/c20250203/250602012/data/math/data/test-00000-of-00001.parquet \
    data.max_response_length=2048 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.project_name="new-AEPO" \
    trainer.save_freq=10 \
    trainer.save_limit=8 \
    trainer.val_freq=-1 \
    trainer.experiment_name="${NAME}" \
    trainer.save_checkpoint_path="/vepfs-mlp2/c20250203/250602012/checkpoints/Llama-3.1-8B-Instruct/${NAME}" \
    trainer.total_episodes=4 \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    trainer.algorithm="AEPO" \

