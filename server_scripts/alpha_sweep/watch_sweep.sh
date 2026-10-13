#!/usr/bin/env bash
# Server-side sentinel: wait for the alpha-sweep PID to exit, then write a marker
# file with the outcome. Started with setsid nohup so it survives the ssh session.
#
#   usage: watch_sweep.sh <pid> <result_json> <log_file> <marker_file>
set -u
PID="$1"
RES="$2"
LOG="$3"
MARK="$4"

while kill -0 "$PID" 2>/dev/null; do
  sleep 30
done

{
  echo "finished_at=$(date -Is)"
  echo "watched_pid=$PID"
  if [ -f "$RES" ]; then
    echo "result_status=OK"
    echo "result_bytes=$(stat -c %s "$RES")"
    if command -v python3 >/dev/null 2>&1; then
      python3 -c "import json,sys; d=json.load(open('$RES')); print('n_groups=%d' % len(d.get('groups',{}))); [print('%s n_images=%d T1=%.6f T2_minus_T1=%+.6f' % (k, g['n_images'], g['T1']['psnr']['image_mean'], g['T2']['delta_psnr_vs_T1']['image_mean'])) for k,g in d.get('groups',{}).items()]; print('elapsed_seconds=%.1f' % d.get('elapsed_seconds', -1))" 2>&1 || echo "result_parse=FAILED"
    fi
  else
    echo "result_status=MISSING"
    echo "result_bytes=0"
  fi
  echo "--- last 20 log lines ---"
  tail -n 20 "$LOG" 2>/dev/null
} > "$MARK" 2>&1

exit 0
