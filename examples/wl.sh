set -x

MODEL_PATH=/home/dataset-assist-0/wc/models/Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

FORMAT_PROMPT="""\nPlease use Matplotlib or seaborn for the drawing. You should make sure that the code can be directly executed without relying on any input data."""

python3 -m verl.trainer.main \
    config=wl_config.yaml \
    data.train_files=WaltonFuture/CodeMLLM1@train \
    data.val_files=WaltonFuture/CodeMLLM1@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=code-7b-new \
    trainer.n_gpus_per_node=8 \

