#!/bin/bash

# 元のファイル名と出力ファイル名
original_file="run_base.sh"
edited_file="run.sh"

# カレントディレクトリを取得
current_dir=$(pwd)

# run.sh を run_edit.sh に変換
sed "s|{CURRENTDIR}|$current_dir|g" "$original_file" > "$edited_file"

# 実行権限を付与
chmod +x "$edited_file"

echo "生成されたスクリプト: $edited_file"