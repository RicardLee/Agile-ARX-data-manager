#!/bin/bash

START_TIME=$(date +%s)
echo ">>> 脚本开始运行: $(date '+%F %T')"

SOURCE_DIR="/home/agilex/data/"
TARGET_DIR="ceph-manipS1:manip_S1/myData-A1/real/raw_data/agilex_split_aloha"
TARGET_DIR_upload="ceph-manipS1:manip_S1/myData-A1/real/raw_data"


TODAY=$(date +%Y-%m-%d)
NEW_FILES=()

# 遍历一级子目录
for folder in "$SOURCE_DIR"*/; do
    folder_name=$(basename "$folder")

    # 排除以 'aloha' 开头的文件夹
    if [[ "$folder_name" == aloha* ]]; then
        continue
    fi

    # 替换文件夹名中的逗号为下划线
    #safe_folder_name=$(echo "$folder_name" | sed -E 's/,_*/__/g' | sed 's/[_\.]\+$//')
    safe_folder_name=$(echo "$folder_name" | sed -E 's/,_*/__/g; s/_+$//')

    # 查找该目录下今天新增的文件
    while IFS= read -r -d $'\0' file; do
        rel_path="${safe_folder_name}/${file#$folder}"  # 构建相对路径
        NEW_FILES+=("$rel_path")
    done < <(find "$folder" -type f -newermt "$TODAY" -print0)
done

NEW_FILES_COUNT=${#NEW_FILES[@]}



CONFIG_PATH="/home/agilex/Desktop/device.config"
DEVICE_ID=$(grep "^device_id:" "$CONFIG_PATH" | awk -F: '{print $2}' | tr -d '[:space:]')

# 创建 JSON 路径
JSON_FILE_DIR="$(dirname "$SOURCE_DIR")/upload_stats"
mkdir -p "$JSON_FILE_DIR"
JSON_FILE_PATH="$JSON_FILE_DIR/upload_stats_$(date +%Y%m%d)_${DEVICE_ID}.json"

# 写 JSON 文件
{
    echo "{"
    echo "  \"date\": \"$TODAY\","
    echo "  \"new_files_count\": $NEW_FILES_COUNT,"
    echo "  \"new_files\": ["
    for ((i=0; i<NEW_FILES_COUNT; i++)); do
        file="${NEW_FILES[$i]}"
        if [ $i -lt $((NEW_FILES_COUNT - 1)) ]; then
            echo "    \"${file}\","
        else
            echo "    \"${file}\""
        fi
    done
    echo "  ]"
    echo "}"
} > "$JSON_FILE_PATH"

echo "当天新增文件数: $NEW_FILES_COUNT"
echo "JSON 文件生成路径: $JSON_FILE_PATH"

# 获取 IP 地址
IP_ADDRESS=$(ip -4 addr show eno1 | grep -oP '(?<=inet\s)\d+\.\d+\.\d+\.\d+')

# 上传数据（只上传符合条件的子目录）
for folder in "$SOURCE_DIR"*/; do
    folder_name=$(basename "$folder")

    if [[ "$folder_name" == aloha* ]]; then
        continue
    fi

    # 替换并清洗目录名
    safe_folder_name=$(echo "$folder_name" | sed -E 's/,_*/__/g; s/_+$//')

    echo "正在上传: $folder_name -> $TARGET_DIR/$safe_folder_name"
    rclone copy "$folder" "$TARGET_DIR/$safe_folder_name" -v --bind $IP_ADDRESS
done

# 上传 JSON 文件
rclone copy -v "$JSON_FILE_PATH" "$TARGET_DIR_upload/upload_stats/" --bind $IP_ADDRESS

# 检查结果
if [ $? -eq 0 ]; then
    echo "文件和统计JSON上传成功！"
else
    echo "上传失败！"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ">>> 脚本运行结束: $(date '+%F %T')，总耗时: ${DURATION} 秒"