
get_device_info() {
  local device_id=$1
  local field=$2
  sed -n "/id: $device_id/,/$field:/p" devices.yaml | grep "$field:" | awk -F': ' '{print $2}'
}

model=$(get_device_info 1 "model")
device_id=$(get_device_info 1 "id")

echo "Device ID: $device_id"
echo "Model: $model"

#!/bin/bash
# conda activate aloha
# python scripts/generate_daily_statistic.py --device_id $1 --model $2