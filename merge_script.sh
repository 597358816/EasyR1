for step in $(seq 40 10 130)
do
    echo ${step}
    python3 /home/dataset-local/EasyR1/scripts/model_merger.py --local_dir /home/dataset-local/checkpoints/Qwen3-4B/qwen3-4b-LP2/global_step_${step}/actor/
done


