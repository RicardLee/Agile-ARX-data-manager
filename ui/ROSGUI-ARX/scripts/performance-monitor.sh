#!/bin/bash

# ROSGUI 性能监控和分析工具
# 提供实时性能监控、回归检测和优化建议

set -e

echo "📊 ROSGUI 性能监控系统"
echo "====================="
echo "实时监控 | 回归检测 | 优化建议"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[监控]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

print_metric() {
    echo -e "${CYAN}[指标]${NC} $1"
}

print_analysis() {
    echo -e "${PURPLE}[分析]${NC} $1"
}

# 性能基线数据
declare -A PERFORMANCE_BASELINES=(
    ["unit_tests_max_time"]="10"
    ["integration_tests_max_time"]="60"
    ["system_tests_max_time"]="180"
    ["benchmark_min_ops_per_sec"]="1000"
    ["memory_usage_max_mb"]="512"
    ["cpu_usage_max_percent"]="80"
)

# 创建性能数据目录
setup_performance_tracking() {
    local perf_dir="performance_data"
    mkdir -p $perf_dir
    
    # 创建性能历史文件
    local timestamp=$(date +%Y%m%d_%H%M%S)
    PERF_LOG="$perf_dir/performance_${timestamp}.log"
    METRICS_FILE="$perf_dir/metrics_${timestamp}.json"
    
    print_info "性能数据将保存到: $perf_dir"
}

# 实时系统监控
start_system_monitoring() {
    print_info "启动实时系统监控..."
    
    # 后台监控进程
    (
        while true; do
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
            local memory_usage=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')
            local memory_mb=$(free -m | awk 'NR==2{print $3}')
            local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
            
            # 记录到日志
            echo "$timestamp,CPU:$cpu_usage,Memory:${memory_usage}%,MemoryMB:$memory_mb,Load:$load_avg" >> $PERF_LOG
            
            # 检查性能阈值
            check_performance_thresholds $cpu_usage $memory_usage $memory_mb
            
            sleep 2
        done
    ) &
    
    MONITOR_PID=$!
    echo $MONITOR_PID > /tmp/rosgui_perf_monitor.pid
    
    print_success "系统监控已启动 (PID: $MONITOR_PID)"
}

# 检查性能阈值
check_performance_thresholds() {
    local cpu_usage=$1
    local memory_percent=$2
    local memory_mb=$3
    
    # CPU使用率检查
    if (( $(echo "$cpu_usage > ${PERFORMANCE_BASELINES[cpu_usage_max_percent]}" | bc -l) )); then
        print_warning "CPU使用率过高: ${cpu_usage}% (阈值: ${PERFORMANCE_BASELINES[cpu_usage_max_percent]}%)"
    fi
    
    # 内存使用检查
    if (( memory_mb > ${PERFORMANCE_BASELINES[memory_usage_max_mb]} )); then
        print_warning "内存使用过高: ${memory_mb}MB (阈值: ${PERFORMANCE_BASELINES[memory_usage_max_mb]}MB)"
    fi
}

# 测试性能分析
analyze_test_performance() {
    print_info "分析测试性能..."

    # 使用绝对路径避免相对路径问题
    local build_dir="/home/qiuzherui/repo/ROSGUI/build"
    if [ ! -d "$build_dir" ]; then
        print_error "构建目录不存在: $build_dir"
        return 1
    fi

    # 简化的性能分析 - 运行make test
    print_metric "运行测试套件..."
    local test_start=$(date +%s)

    (
        cd "$build_dir"
        if make test > /tmp/test_performance_output.log 2>&1; then
            echo "SUCCESS"
        else
            echo "FAILED"
        fi
    ) > /tmp/test_result.log

    local test_end=$(date +%s)
    local test_duration=$((test_end - test_start))

    if grep -q "SUCCESS" /tmp/test_result.log; then
        print_metric "测试总耗时: ${test_duration}秒"

        if (( test_duration > ${PERFORMANCE_BASELINES[unit_tests_max_time]} )); then
            print_warning "测试耗时较长: ${test_duration}s"
        else
            print_success "测试性能良好: ${test_duration}s"
        fi
    else
        print_error "测试执行失败"
        return 1
    fi
}

# 基准测试性能分析
analyze_benchmark_performance() {
    print_metric "分析基准测试性能..."
    
    # 运行基准测试并捕获输出
    if [ -f "./bin/tools/benchmark_modernized_production" ]; then
        local benchmark_output=$(timeout 60 ./bin/tools/benchmark_modernized_production 2>&1 || true)
        
        # 解析基准测试结果
        local ops_per_sec=$(echo "$benchmark_output" | grep -o '[0-9]\+\.[0-9]\+ ops/sec' | head -1 | awk '{print $1}')
        
        if [ -n "$ops_per_sec" ]; then
            print_metric "基准测试性能: ${ops_per_sec} ops/sec"
            
            if (( $(echo "$ops_per_sec < ${PERFORMANCE_BASELINES[benchmark_min_ops_per_sec]}" | bc -l) )); then
                print_warning "基准测试性能低于基线: ${ops_per_sec} < ${PERFORMANCE_BASELINES[benchmark_min_ops_per_sec]}"
            else
                print_success "基准测试性能良好: ${ops_per_sec} ops/sec"
            fi
        else
            print_warning "无法解析基准测试结果"
        fi
    else
        print_warning "基准测试可执行文件不存在"
    fi
}

# 生成性能报告
generate_performance_report() {
    print_info "生成性能分析报告..."
    
    local report_file="performance_analysis_report.md"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat > $report_file << EOF
# ROSGUI 性能分析报告

**生成时间**: $timestamp  
**系统**: $(uname -a)  
**CPU**: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)  
**内存**: $(free -h | awk 'NR==2{print $2}')  

## 📊 性能指标总览

### 测试执行时间
EOF

    # 添加测试时间分析
    if [ -f "/tmp/unit_test_output.log" ]; then
        echo "- **单元测试**: $(grep "tests passed" /tmp/unit_test_output.log | tail -1 || echo "数据不可用")" >> $report_file
    fi
    
    if [ -f "/tmp/integration_test_output.log" ]; then
        echo "- **集成测试**: $(grep "tests passed" /tmp/integration_test_output.log | tail -1 || echo "数据不可用")" >> $report_file
    fi
    
    # 添加系统资源使用情况
    cat >> $report_file << EOF

### 系统资源使用

EOF
    
    if [ -f "$PERF_LOG" ]; then
        echo "#### 监控期间资源使用情况" >> $report_file
        echo '```' >> $report_file
        echo "时间,CPU使用率,内存使用率,内存(MB),负载" >> $report_file
        tail -20 $PERF_LOG >> $report_file
        echo '```' >> $report_file
    fi
    
    # 添加性能建议
    cat >> $report_file << EOF

## 🎯 性能优化建议

### 基于当前分析的建议:

1. **编译优化**
   - 使用 \`-O3 -march=native\` 进行最大优化
   - 启用 LTO (Link Time Optimization)
   - 考虑使用 PGO (Profile Guided Optimization)

2. **测试执行优化**
   - 并行执行单元测试以减少总时间
   - 使用 CPU 亲和性绑定测试到特定核心
   - 实施测试结果缓存机制

3. **内存优化**
   - 监控内存泄漏和过度分配
   - 使用内存池减少动态分配开销
   - 优化数据结构布局提高缓存命中率

4. **系统级优化**
   - 调整系统调度器参数
   - 禁用不必要的系统服务
   - 使用高性能文件系统 (如 tmpfs) 进行临时文件

## 📈 历史趋势

$(ls performance_data/*.log 2>/dev/null | wc -l) 个历史性能记录可用于趋势分析。

## 🔧 下次运行建议

- 在相同系统负载下重复测试以确保一致性
- 考虑在不同硬件配置下进行基准测试
- 实施自动化性能回归检测

---
*报告由 ROSGUI 性能监控系统自动生成*
EOF

    print_success "性能报告已生成: $report_file"
}

# 停止监控
stop_monitoring() {
    if [ -f "/tmp/rosgui_perf_monitor.pid" ]; then
        local pid=$(cat /tmp/rosgui_perf_monitor.pid)
        kill $pid 2>/dev/null || true
        rm -f /tmp/rosgui_perf_monitor.pid
        print_info "性能监控已停止"
    fi
}

# 清理临时文件
cleanup() {
    stop_monitoring
    rm -f /tmp/unit_test_output.log /tmp/integration_test_output.log
}

# 主执行流程
main() {
    local action="monitor"
    local duration=60
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --analyze)
                action="analyze"
                shift
                ;;
            --duration)
                duration="$2"
                shift 2
                ;;
            --report)
                action="report"
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --analyze           执行性能分析"
                echo "  --duration <秒>     监控持续时间 (默认: 60)"
                echo "  --report            生成性能报告"
                echo "  --help              显示帮助信息"
                exit 0
                ;;
            *)
                print_warning "未知选项: $1"
                shift
                ;;
        esac
    done
    
    # 设置清理陷阱
    trap cleanup EXIT
    
    # 初始化性能跟踪
    setup_performance_tracking
    
    case $action in
        "monitor")
            print_info "启动性能监控 (持续时间: ${duration}秒)"
            start_system_monitoring
            sleep $duration
            stop_monitoring
            print_success "监控完成"
            ;;
        "analyze")
            print_info "执行完整性能分析"
            start_system_monitoring
            analyze_test_performance
            sleep 5  # 让监控收集一些数据
            stop_monitoring
            generate_performance_report
            ;;
        "report")
            generate_performance_report
            ;;
    esac
    
    print_success "性能监控任务完成"
}

# 执行主流程
main "$@"
