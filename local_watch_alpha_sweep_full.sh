#!/usr/bin/env bash
# Local watchdog: poll the remote marker file; when the sweep finishes, download
# the result and generate the Chinese markdown + compact JSON report.
# Run in background; the agent is notified when this exits.
set -u

SSH_HOST="li@10.177.19.114"
RDIR="/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/evaluation_872/alpha_sweep"
LDIR="/f/WORK/projects/pidsr_recurrent_joint_k2/_dl/alpha_sweep"
PROJ="/f/WORK/projects/pidsr_recurrent_joint_k2"
PY="C:/Users/14030/.workbuddy/binaries/python/versions/3.13.12/python.exe"
MARK="alpha_sweep_full.DONE"
RES="alpha_sweep_full_result.json"

mkdir -p "$LDIR"
echo "WATCH_START $(date -Is)  remote=$RDIR/$MARK"

while true; do
  if ssh -o ConnectTimeout=20 -o BatchMode=yes "$SSH_HOST" "test -f $RDIR/$MARK" 2>/dev/null; then
    echo "MARKER_FOUND $(date -Is)"
    break
  fi
  sleep 120
done

echo "--- remote marker ---"
ssh -o ConnectTimeout=30 "$SSH_HOST" "cat $RDIR/$MARK" 2>&1

echo "--- downloading ---"
scp -o ConnectTimeout=60 "$SSH_HOST:$RDIR/$RES" "$LDIR/" 2>&1
ls -l "$LDIR/$RES" 2>&1

echo "--- generating report ---"
"$PY" "$PROJ/server_scripts/alpha_sweep/summarize_alpha_sweep.py" \
  "$LDIR/$RES" \
  --out-md "$PROJ/analysis/ALPHA_SWEEP_FULL_VALIDATION_ZH_20261002.md" \
  --out-json "$PROJ/analysis/ALPHA_SWEEP_FULL_VALIDATION_ZH_20261002_summary.json" 2>&1

echo "LOCAL_PIPELINE_DONE $(date -Is)"
