for step in $(seq 110 10 300)
do
    echo ${step}
    python3 scripts/model_merger.py --local_dir  /home/dataset-assist-0/wc/EasyR1-main/examples/checkpoints/Entropy-Controller/IS-e0.5-a0.15-noratio/global_step_${step}/actor/
done


