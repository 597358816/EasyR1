for step in $(seq 10 10 50)
do
    echo ${step}
    python3 /home/dataset-local/EasyR1/scripts/model_merger.py --local_dir /home/dataset-local/checkpoints/DSQW-7B/dsqw-7b-LP2/global_step_${step}/actor/
done


