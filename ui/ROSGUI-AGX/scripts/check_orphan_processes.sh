#!/bin/bash
# ROSGUI进程残留检测和清理脚本

echo "检查ROSGUI相关的残留进程..."

# 检查roscore进程
ROSCORE_PIDS=$(pgrep -f "roscore" || true)
if [ -n "$ROSCORE_PIDS" ]; then
    echo "发现roscore进程: $ROSCORE_PIDS"
    for pid in $ROSCORE_PIDS; do
        echo "  PID $pid: $(ps -p $pid -o cmd --no-headers)"
    done
else
    echo "✓ 未发现roscore残留进程"
fi

# 检查rosmaster进程
ROSMASTER_PIDS=$(pgrep -f "rosmaster" || true)
if [ -n "$ROSMASTER_PIDS" ]; then
    echo "发现rosmaster进程: $ROSMASTER_PIDS"
    for pid in $ROSMASTER_PIDS; do
        echo "  PID $pid: $(ps -p $pid -o cmd --no-headers)"
    done
else
    echo "✓ 未发现rosmaster残留进程"
fi

# 检查roslaunch进程
ROSLAUNCH_PIDS=$(pgrep -f "roslaunch" || true)
if [ -n "$ROSLAUNCH_PIDS" ]; then
    echo "发现roslaunch进程: $ROSLAUNCH_PIDS"
    for pid in $ROSLAUNCH_PIDS; do
        echo "  PID $pid: $(ps -p $pid -o cmd --no-headers)"
    done
else
    echo "✓ 未发现roslaunch残留进程"
fi

# 检查ROS节点进程（一般以python开头，包含ROS相关路径）
ROS_NODE_PIDS=$(pgrep -f "python.*ros" || true)
if [ -n "$ROS_NODE_PIDS" ]; then
    echo "发现ROS节点进程: $ROS_NODE_PIDS"
    for pid in $ROS_NODE_PIDS; do
        echo "  PID $pid: $(ps -p $pid -o cmd --no-headers | cut -c1-100)..."
    done
else
    echo "✓ 未发现ROS节点残留进程"
fi

# 检查ROSGUI-AGX进程
ROSGUI_PIDS=$(pgrep -f "ROSGUIAGX" || true)
if [ -n "$ROSGUI_PIDS" ]; then
    echo "发现ROSGUI-AGX进程: $ROSGUI_PIDS"
    for pid in $ROSGUI_PIDS; do
        echo "  PID $pid: $(ps -p $pid -o cmd --no-headers)"
    done
else
    echo "✓ 未发现ROSGUI-AGX残留进程"
fi

# 检查端口占用
ROS_PORT_CHECK=$(netstat -tlnp 2>/dev/null | grep ":11311 " || true)
if [ -n "$ROS_PORT_CHECK" ]; then
    echo "⚠ ROS端口11311仍被占用:"
    echo "$ROS_PORT_CHECK"
else
    echo "✓ ROS端口11311未被占用"
fi

# 提供清理选项
if [ -n "$ROSCORE_PIDS" ] || [ -n "$ROSMASTER_PIDS" ] || [ -n "$ROSLAUNCH_PIDS" ] || [ -n "$ROS_NODE_PIDS" ] || [ -n "$ROSGUI_PIDS" ]; then
    echo ""
    echo "是否要清理这些残留进程? (y/N)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo "正在清理残留进程..."
        
        # 优雅终止
        for pid in $ROSCORE_PIDS $ROSMASTER_PIDS $ROSLAUNCH_PIDS $ROS_NODE_PIDS $ROSGUI_PIDS; do
            if kill -TERM "$pid" 2>/dev/null; then
                echo "  发送SIGTERM到进程 $pid"
            fi
        done
        
        sleep 3  # 给roslaunch更多时间来清理其子进程
        
        # 强制终止仍在运行的进程
        for pid in $ROSCORE_PIDS $ROSMASTER_PIDS $ROSLAUNCH_PIDS $ROS_NODE_PIDS $ROSGUI_PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  强制终止进程 $pid"
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        
        echo "清理完成"
    fi
fi

echo "检查完成" 