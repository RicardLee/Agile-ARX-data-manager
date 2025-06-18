#!/bin/bash
set -euo pipefail

# 定义获取设备信息的函数
get_device_info() {
    local yaml_file="$1"
    local key="${2:-ip}"

    # 验证输入参数
    [[ $# -lt 1 ]] && echo "Usage: ${FUNCNAME[0]} <yaml_file> [key]" >&2 && return 1
    [[ ! -f "$yaml_file" ]] && echo "Error: YAML file not found: $yaml_file" >&2 && return 1
    [[ ! "$key" =~ ^(id|ip|model)$ ]] && echo "Error: Invalid key '$key'. Use 'id'/'ip'/'model'" >&2 && return 1

    # 检查yq依赖
    if ! command -v yq &>/dev/null; then
        echo "Error: yq command not found. Install with: pip install yq" >&2
        return 1
    fi

    local result
    if ! result=$(yq -r ".device_info[].${key}" "$yaml_file" 2>/dev/null | head -n1); then
        echo "Error: Failed to parse YAML" >&2
        return 1
    fi

    [[ -z "$result" ]] && echo "Error: No '$key' field found" >&2 && return 1
    echo "$result"
}

# 安全激活conda环境
activate_conda() {
    # 检查是否已经在目标环境
    if [[ -n "$CONDA_DEFAULT_ENV" && "$CONDA_DEFAULT_ENV" == "aloha" ]]; then
        echo "Already in conda environment: aloha"
        return 0
    fi

    # 检查conda是否可用
    if ! command -v conda &>/dev/null; then
        echo "Error: conda command not found" >&2
        return 1
    fi

    # 初始化conda（如果尚未初始化）
    if [[ -z "$CONDA_SHLVL" ]]; then
        __conda_setup="$('/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
        eval "$__conda_setup" || {
            echo "Error: Conda initialization failed" >&2
            return 1
        }
    fi

    # 尝试激活环境
    if ! conda activate aloha 2>/dev/null; then
        echo "Error: Failed to activate 'aloha' environment. Available environments:" >&2
        conda env list >&2
        return 1
    fi
    echo "Successfully activated conda environment: aloha"
}

main() {
    # 获取设备信息
    local device_id model
    device_id=$(get_device_info "devices.yaml" "id") || exit 1
    model=$(get_device_info "devices.yaml" "model") || exit 1

    # 激活conda环境
    if ! activate_conda; then
        echo "Proceeding without conda environment activation" >&2
    fi

    # 主业务逻辑
    echo "  Device Info:"
    echo "  ID: $device_id"
    echo "  Model: $model"

    python scripts/calculate_time_duration_single.py ~/data $device_id $model
}

main "$@"
