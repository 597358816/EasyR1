MODEL_NAME="Qwen3-4B"
RUN_NAME="qwen3-4b-KLcov"

#for step in $(seq 200 10 270)
#for step in $(seq 50 10 120)
#for step in $(seq 40 10 150)
#for step in $(seq 60 10 130)
for step in $(seq 60 10 200)
do
    echo ${step}
    python3 ./scripts/model_merger.py --local_dir "/home/dataset-assist-0/wc/checkpoints/${MODEL_NAME}/${RUN_NAME}/global_step_${step}/actor/"
done

#rclone copy "/home/dataset-assist-0/wc/checkpoints/${MODEL_NAME}/${RUN_NAME}" "beijing11:bucket-c20250203/wc/checkpoints/${MODEL_NAME}/${RUN_NAME}"  --progress --transfers=48

