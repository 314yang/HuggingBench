#!/bin/bash

# 定义模型和数据集列表
models=("rescal" "distmult" "conve" "complex" "transformer")
# models=("distmult" "transformer")

# 遍历所有模型和数据集组合
for model in "${models[@]}"; do
    echo "Running model: $model "
    kge start examples/huggingface-train-$model.yaml --job.device cuda:3
    if [ $? -eq 0 ]; then
        echo "Successfully executed model: $model"
    else
        echo "Failed to execute model: $model"
    fi
    echo "----------------------------------------"
done

echo "All combinations executed."