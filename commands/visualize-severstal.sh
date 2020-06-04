#!/usr/bin/env bash

cd /mnt/Work/segmenter

conda activate segmenter

set -euo pipefail

# Strong Learner
launch collect --collector metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch collect --collector predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch collect --collector instance-metrics --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch visualize --visualizer metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch visualize --visualizer auc --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch visualize --visualizer predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch visualize --visualizer instance-metrics --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch collect --collector layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch visualize --visualizer layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch visualize --visualizer best-threshold --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
launch collect --collector confusion --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1

launch collect --collector metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch collect --collector predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch collect --collector instance-metrics --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch visualize --visualizer metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch visualize --visualizer auc --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch visualize --visualizer predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch visualize --visualizer instance-metrics --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch collect --collector layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch visualize --visualizer layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch visualize --visualizer best-threshold --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
launch collect --collector confusion --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2

launch collect --collector metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch collect --collector predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch collect --collector instance-metrics --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch visualize --visualizer metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch visualize --visualizer auc --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch visualize --visualizer predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch visualize --visualizer instance-metrics --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch collect --collector layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch visualize --visualizer layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch visualize --visualizer best-threshold --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
launch collect --collector confusion --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3

launch collect --collector metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch collect --collector predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch collect --collector instance-metrics --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch visualize --visualizer metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch visualize --visualizer auc --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch visualize --visualizer predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch visualize --visualizer instance-metrics --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch collect --collector layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch visualize --visualizer layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch visualize --visualizer best-threshold --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
launch collect --collector confusion --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4

launch visualize --visualizer confusion --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal

# Baseline
launch collect --collector metric --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch collect --collector predict --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch collect --collector instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch visualize --visualizer metric --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch visualize --visualizer auc --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch visualize --visualizer predict --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch visualize --visualizer instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch collect --collector layer-output --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch visualize --visualizer layer-output --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch visualize --visualizer best-threshold --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1
launch collect --collector confusion --job-config c041f9d41605254f805232999f143ab0 severstal --classes 1

launch collect --collector metric --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch collect --collector predict --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch collect --collector instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch visualize --visualizer metric --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch visualize --visualizer auc --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch visualize --visualizer predict --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch visualize --visualizer instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch collect --collector layer-output --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch visualize --visualizer layer-output --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch visualize --visualizer best-threshold --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2
launch collect --collector confusion --job-config c041f9d41605254f805232999f143ab0 severstal --classes 2

launch collect --collector metric --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch collect --collector predict --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch collect --collector instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch visualize --visualizer metric --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch visualize --visualizer auc --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch visualize --visualizer predict --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch visualize --visualizer instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch collect --collector layer-output --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch visualize --visualizer layer-output --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch visualize --visualizer best-threshold --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3
launch collect --collector confusion --job-config c041f9d41605254f805232999f143ab0 severstal --classes 3

launch collect --collector metric --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch collect --collector predict --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch collect --collector instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch visualize --visualizer metric --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch visualize --visualizer auc --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch visualize --visualizer predict --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch visualize --visualizer instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch collect --collector layer-output --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch visualize --visualizer layer-output --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch visualize --visualizer best-threshold --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4
launch collect --collector confusion --job-config c041f9d41605254f805232999f143ab0 severstal --classes 4

launch visualize --visualizer confusion --job-config c041f9d41605254f805232999f143ab0 severstal

# Boosted Learner
launch collect --collector metric --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch collect --collector predict --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch collect --collector instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch visualize --visualizer metric --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch visualize --visualizer auc --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch visualize --visualizer predict --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch visualize --visualizer instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch collect --collector layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch visualize --visualizer layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch visualize --visualizer best-threshold --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1
launch collect --collector confusion --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 1

launch collect --collector metric --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch collect --collector predict --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch collect --collector instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch visualize --visualizer metric --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch visualize --visualizer auc --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch visualize --visualizer predict --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch visualize --visualizer instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch collect --collector layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch visualize --visualizer layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch visualize --visualizer best-threshold --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2
launch collect --collector confusion --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 2

launch collect --collector metric --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch collect --collector predict --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch collect --collector instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch visualize --visualizer metric --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch visualize --visualizer auc --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch visualize --visualizer predict --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch visualize --visualizer instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch collect --collector layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch visualize --visualizer layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch visualize --visualizer best-threshold --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3
launch collect --collector confusion --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 3

launch collect --collector metric --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch collect --collector predict --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch collect --collector instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch visualize --visualizer metric --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch visualize --visualizer auc --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch visualize --visualizer predict --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch visualize --visualizer instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch collect --collector layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch visualize --visualizer layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch visualize --visualizer best-threshold --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4
launch collect --collector confusion --job-config c4fc907d76c4adc975169e34d32b95df severstal --classes 4

launch visualize --visualizer confusion --job-config c4fc907d76c4adc975169e34d32b95df severstal

# Weak Learner
launch collect --collector metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch collect --collector predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch collect --collector instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch visualize --visualizer metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch visualize --visualizer auc --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch visualize --visualizer predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch visualize --visualizer instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch collect --collector layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch visualize --visualizer layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch visualize --visualizer best-threshold --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1
launch visualize --collector confusion --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 1

launch collect --collector metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch collect --collector predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch collect --collector instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch visualize --visualizer metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch visualize --visualizer auc --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch visualize --visualizer predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch visualize --visualizer instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch collect --collector layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch visualize --visualizer layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch visualize --visualizer best-threshold --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2
launch visualize --collector confusion --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 2

launch collect --collector metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch collect --collector predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch collect --collector instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch visualize --visualizer metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch visualize --visualizer auc --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch visualize --visualizer predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch visualize --visualizer instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch collect --collector layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch visualize --visualizer layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch visualize --visualizer best-threshold --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3
launch visualize --collector confusion --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 3

launch collect --collector metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch collect --collector predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch collect --collector instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch visualize --visualizer metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch visualize --visualizer auc --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch visualize --visualizer predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch visualize --visualizer instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch collect --collector layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch visualize --visualizer layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch visualize --visualizer best-threshold --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4
launch visualize --collector confusion --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal --classes 4

launch visualize --visualizer confusion --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal