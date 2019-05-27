#! /usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
srun -p VI_Face_1080TI --mpi=pmi2 --gres=gpu:8 --ntasks-per-node=1  -n1  --job-name=cml --kill-on-bad-exit=1  python -u main.py > log/train_$now.log 2>&1 &
