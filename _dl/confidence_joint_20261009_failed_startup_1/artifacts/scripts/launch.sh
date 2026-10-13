#!/usr/bin/env bash
set -euo pipefail
source /media/li/data3/25thstudent/DY/929work/activate-pidsr-k2.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd /media/li/data3/25thstudent/DY/929work/pidsr_k2_929/confidence_joint_20261009
mkdir -p logs
python -u scripts/prepare.py
nohup python -u scripts/run_overnight.py > logs/overnight.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/overnight.pid
cat logs/overnight.pid
