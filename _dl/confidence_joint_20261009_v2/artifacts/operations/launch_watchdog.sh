#!/usr/bin/env bash
set -euo pipefail
cd /media/li/data3/25thstudent/DY/929work/pidsr_k2_929/confidence_joint_20261009_v2
nohup /usr/bin/python3 -u operations/watch_deadline.py > operations/watchdog.log 2>&1 < /dev/null &
printf '%s\n' "$!" > operations/watchdog.pid
cat operations/watchdog.pid
