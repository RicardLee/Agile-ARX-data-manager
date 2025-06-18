#!/bin/bash
REPO_DIR="/home/agilex/Desktop/Agile-ARX-data-manager" # 替换为实际路径
DATA_DIR="$REPO_DIR/data_statistics/" # 数据文件夹
REMOTE="origin" # 远程仓库名称
BRANCH="main" # 分支名称

cd "$REPO_DIR" || exit 1

if ! git status --porcelain "$DATA_DIR" | grep -q .; then
    echo "No changes detected in $DATA_DIR. Exiting."
    exit 0
fi

git add "$DATA_DIR"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
git commit -m "Auto-push new files from $DATA_DIR at $TIMESTAMP"

if git push "$REMOTE" "$BRANCH"; then
    echo "Successfully pushed to $REMOTE/$BRANCH"
else
    echo "Failed to push to $REMOTE/$BRANCH"
    exit 1
fi