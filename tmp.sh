# for step in $(seq 160 20 300)
# do
#     echo ${step}
#     python3 /vepfs-mlp2/c20250203/250602012/EasyR1/scripts/model_merger.py --local_dir /vepfs-mlp2/c20250203/250602012/checkpoints/Qwen2.5-math-7B/qwen-math-7b-GRPO-add/global_step_${step}/actor/
# done

python3 /vepfs-mlp2/c20250203/250602012/EasyR1/scripts/model_merger.py --local_dir /vepfs-mlp2/c20250203/250602012/checkpoints/Qwen3-8B/qwen3-8b-GRPO/global_step_139/actor/

