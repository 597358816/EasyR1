for step in $(seq 40 10 150)
do
    echo ${step}
    python3 /home/dataset-local/EasyR1/scripts/model_merger.py --local_dir /home/dataset-local/checkpoints/DSQW-7B/dsqw-7b-shorterbetter/global_step_${step}/actor/
done


