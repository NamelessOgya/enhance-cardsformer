#!/bin/bash
set -e

# もし同名コンテナが残っていれば消す
docker rm -f cardsformer_env_container 2>/dev/null || true

# バックグラウンドで立ち上げ。tail で常駐させる
docker run -d \
  --gpus all \
  --name cardsformer_env_container \
  -v "$(pwd)/..:/app" \
  --restart unless-stopped \
  namelessogya/cardsformer_env \
  bash -c "cd ./cf_implement_simple_model/Cardsformer && ./experiment/train_simple_model/run.sh" #実行コマンドに応じて変更

