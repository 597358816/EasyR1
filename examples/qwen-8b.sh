set -x

MODEL_PATH=/home/dataset-assist-0/wc/models/Qwen/Qwen3-8B  # replace it with your local file path


FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer. 
    The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""
# FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer. 
# The final answer MUST BE put in \boxed{}. /no_think"""

python3 -m verl.trainer.main \
    config=config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.train_files=/home/dataset-assist-0/wc/data/dapo/train_dapo.parquet \
    data.val_files=hiyouga/math12k@test \
    data.max_response_length=8192 \
    data.rollout_batch_size=128 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.rollout.n=8 \
    worker.rollout.max_num_batched_tokens=10240 \
    trainer.experiment_name=qwen3-8b-GRPO \
    trainer.val_freq=-1 \
    trainer.save_limit=20 \
    trainer.save_freq=10 \
    trainer.total_episodes=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.actor.global_batch_size=64 \
    #trainer.load_checkpoint_path=/home/dataset-assist-0/wc/EasyR1-main/examples/checkpoints/easyr1/qwen3-4b-AEPO-0.25/global_step_140 \

    # data.rollout_batch_size=64 \
    # worker.actor.global_batch_size=32 \
    # worker.actor.micro_batch_size_per_device_for_update=1 \
    # worker.actor.micro_batch_size_per_device_for_experience=2 \
    # trainer.total_episodes=2 \
    # trainer.save_freq=20 \
