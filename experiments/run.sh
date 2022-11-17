# name: fmnist-resnet18-fsmap
# method: grid
# parameters:
#   lr:
#     values:
#       - 0.1
#   func_decay:
#     values:
#       - 2e-5
#   seed:
#     values:
#       - 42
# program: experiments/train_fsmap.py
# command:
#   - ${env}
#   - ${interpreter}
#   - ${program}
#   - --dataset=fmnist
#   - --batch-size=128
#   - --optimizer=sgd
#   - --epochs=200
#   - ${args}


eval $(python3 -m setGPU)

# run_name='b100_psmap'
# OMP_NUM_THREADS=4 python3 experiments/train.py \
#     --run_name=${run_name} \
#     --log_dir=logs/${run_name} \
#     --model_name=tinycnn \
#     --dataset=fmnist \
#     --batch-size=100 \
#     --optimizer=sgd \
#     --epochs=200 \
#     --lr=0.01 \
#     --weight-decay=2e-5 \
#     --seed=42

run_name='b100_exact_wd2e-5'
OMP_NUM_THREADS=4 python3 experiments/train_exact_fsmap.py \
    --run_name=${run_name} \
    --log_dir=logs/${run_name} \
    --ckpt_path=logs/${run_name} \
    --model_name=tinycnn \
    --dataset=fmnist \
    --batch-size=100 \
    --optimizer=sgd \
    --epochs=200 \
    --lr=0.01 \
    --weight-decay=2e-5 \
    --jitter=1e-5 \
    --seed=42