#!/bin/bash
STRATEGIES=("tetris")
BASE_OUTPUT_DIR="logs"

mkdir -p $BASE_OUTPUT_DIR

# 检查并终止遗留进程
cleanup_previous_processes() {
    echo "清理磁盘持久化文件..."
    rm -rf evicted_params
    rm -rf persistent_cache
    
    echo "正在清理遗留进程..."
    # 清理 proxy.pid 文件中的进程
    if [ -f proxy.pid ]; then
        pid=$(cat proxy.pid)
        if ps -p $pid > /dev/null; then
            echo "终止 proxy 进程 (PID: $pid)..."
            kill -9 $pid
        fi
        rm proxy.pid
    fi
    
    # 清理 python proxy.py 进程
    proxy_pids=$(pgrep -f "python proxy.py")
    if [ ! -z "$proxy_pids" ]; then
        echo "终止 proxy.py 进程 (PIDs: $proxy_pids)..."
        kill -9 $proxy_pids
    fi
    
    sleep 2
    
    # 验证是否还有遗留进程
    remaining_pids=$(pgrep -f "python proxy.py")
    if [ ! -z "$remaining_pids" ]; then
        echo "警告：仍有遗留进程 (PIDs: $remaining_pids)"
        echo "尝试最后一次强制清理..."
        kill -9 $remaining_pids
        sleep 1
    fi
    
    echo "进程清理完成"
}

trap cleanup_previous_processes EXIT

start_proxy() {
    local strategy=$1
    export CACHE_STRATEGY=$strategy
    export USE_PERSISTENT_CACHE=false
    python proxy.py &
    sleep 5  # 等待服务器启动
    echo $! > proxy.pid
}

stop_proxy() {
    if [ -f proxy.pid ]; then
        kill $(cat proxy.pid)
        rm proxy.pid
        sleep 2  # 等待进程完全终止
    fi
}

run_workload() {
    local strategy=$1
    echo "执行 $strategy 策略..."
    mkdir -p "$BASE_OUTPUT_DIR/$strategy"
    python model_load_tetris.py --use-trace --family all --tetris
    sleep 2
}

for strategy in "${STRATEGIES[@]}"; do
    cleanup_previous_processes
    echo "开始测试 $strategy 策略..."
    
    mkdir -p "$BASE_OUTPUT_DIR/$strategy"
    start_proxy $strategy
    run_workload $strategy
    stop_proxy
    
    echo "$strategy 策略测试完成"
    echo "----------------------------"
done

echo "所有实验完成！" 