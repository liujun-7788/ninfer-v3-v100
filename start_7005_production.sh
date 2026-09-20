#!/bin/bash
# 7005 production (2026-09-18, boss order): cloned from start_7106_production.sh
# diff vs 7106: port 7106->7005, max-concurrency 2->1, device-state-slots 2->1, new log file
# rollback to old 7106: bash start_7106_production.sh (conc2/port7106)
cd /data/deploy/ninfer-test
nohup ./build-v100/apps/ninfer-serve /data/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --port 7005 --api-key sk-tHEjIjWjRSyJG4kN6YQKTsVFuaAEQH4NNxDqjFN5zgbRGJRp \
  --device 1 --model-id qwen3.8-27b --max-context 230000 --prefill-chunk 2048 \
  --kv-capacity auto --max-concurrency 1 --kv-dtype int8 --device-state-slots 1 \
  --host-state-slots 8 --host-kv-mib 8192 --spec mtp --draft-tokens 3 \
  --lm-head-draft --preserve-thinking --pending-timeout-ms 600000 \
  --vision --log-level info > v3_int8_230k_7005.log 2>&1 &
echo "STARTED pid=$!"
