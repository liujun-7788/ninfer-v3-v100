#!/bin/bash
# 7006 production (2026-09-18, boss order): clone of 7005 config on GPU2
# diff vs 7005: port 7005->7006, device 1->2, new log file
cd /data/deploy/ninfer-test
nohup ./build-v100/apps/ninfer-serve /data/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --port 7006 --api-key sk-tHEjIjWjRSyJG4kN6YQKTsVFuaAEQH4NNxDqjFN5zgbRGJRp \
  --device 2 --model-id qwen3.8-27b --max-context 230000 --prefill-chunk 2048 \
  --kv-capacity auto --max-concurrency 1 --kv-dtype int8 --device-state-slots 1 \
  --host-state-slots 8 --host-kv-mib 8192 --spec mtp --draft-tokens 3 \
  --lm-head-draft --preserve-thinking --pending-timeout-ms 600000 \
  --vision --log-level info > v3_int8_230k_7006.log 2>&1 &
echo "STARTED pid=$!"
