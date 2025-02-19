#!/bin/bash

# 定义模型和数据集列表
models=("kgcl")
datasets=("huggingface_2hop")

# 遍历所有模型和数据集组合
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running model: $model with dataset: $dataset"
        python ./main.py --model "$model" --dataset "$dataset" --cuda "2"
        if [ $? -eq 0 ]; then
            echo "Successfully executed model: $model with dataset: $dataset"
        else
            echo "Failed to execute model: $model with dataset: $dataset"
        fi
        echo "----------------------------------------"
    done
done

echo "All combinations executed."