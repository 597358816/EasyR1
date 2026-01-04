for step in $(seq 40 10 130)
do
    echo ${step}
    python3 /home/dataset-assist-0/wc/EasyR1-main/scripts/model_merger.py --local_dir /home/dataset-assist-0/wc/EasyR1-main/examples/checkpoints/easyr1/qwen3-4b-AEPO-0.5-2/global_step_${step}/actor/
done


