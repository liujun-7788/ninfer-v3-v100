#!/usr/bin/env bash
set -u
LOG=/data/deploy/ninfer-test/q8build.log
echo "=== q8simt build restart $(date) ===" >> "$LOG"
touch /data/deploy/ninfer-test/src/ops/weight_input.cpp
cd /data/deploy/ninfer-test
bash do_q8build.sh >> "$LOG" 2>&1
