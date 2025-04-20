#!/bin/bash

get_params() {
  local strategy=$1
  case "$strategy" in
    "PiSeL")
      echo "--async-load"
      ;;
    "Mini")
      echo "--async-load --mini-loader"
      ;;
    "Preload")
      echo "--async-load --pre-load"
      ;;
    "Cicada")
      echo "--async-load --pre-load --mini-loader"
      ;;
    *)
      echo ""
      ;;
  esac
}

strategies=("PiSeL" "Mini" "Preload" "Cicada")

for strategy in "${strategies[@]}"; do
  params=$(get_params "$strategy")
  echo "执行策略: $strategy, 参数: $params"
  python3 pipeline_model.py $params
  sleep 3
done
bash run_experiments_tetris.sh
bash run_memory_comparison.sh

python3 analyze_performance.py