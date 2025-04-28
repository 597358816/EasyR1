set -x

MODEL_PATH=/home/dataset-assist-0/wc/models/Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

FORMAT_PROMPT="""\nPlease reason step by step, and put your final answer within \boxed{}."""

python3 -m verl.trainer.main \
    config=wl2_config.yaml \
    data.train_files=WaltonFuture/MMMT-ThinkLite-3k-random@train \
    data.val_files=WaltonFuture/mathvision-train@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=wl_replaybuffer \
    trainer.n_gpus_per_node=8
