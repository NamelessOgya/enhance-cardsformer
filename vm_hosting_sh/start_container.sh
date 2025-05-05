#!/bin/bash
set -e

# もし同名コンテナが残っていれば消す
docker rm -f cardsformer_env_container 2>/dev/null || true

# バックグラウンドで立ち上げ。tail で常駐させる
docker run -d \
  --gpus all \
  --name cardsformer_env_container \
  -v "$(pwd):/app" \
  -w "/app" \
  --restart unless-stopped \
  namelessogya/cardsformer_env \
  tail -f /dev/null

exec docker exec -it cardsformer_env_container bash