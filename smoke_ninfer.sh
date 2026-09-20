#!/bin/bash
# smoke test a ninfer instance: GET /v1/models + one small chat completion
PORT=$1
KEY="sk-tHEjIjWjRSyJG4kN6YQKTsVFuaAEQH4NNxDqjFN5zgbRGJRp"
echo "--- GET /v1/models on $PORT ---"
curl -s -m 10 -H "Authorization: Bearer $KEY" http://127.0.0.1:$PORT/v1/models | head -c 400
echo
echo "--- POST /v1/chat/completions on $PORT ---"
curl -s -m 90 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"reply with exactly: OK"}],"max_tokens":64,"temperature":0}' \
  http://127.0.0.1:$PORT/v1/chat/completions | head -c 800
echo
echo "--- smoke done on $PORT ---"
