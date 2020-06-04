#!/usr/bin/env bash

set -euo pipefail

# launch visualize --visualizer best-threshold --job-config c041f9d41605254f805232999f143ab0 severstal
# launch visualize --visualizer best-threshold --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch visualize --visualizer best-threshold --job-config c4fc907d76c4adc975169e34d32b95df severstal

# Baseline
# launch evaluate --evaluator metric --job-config c041f9d41605254f805232999f143ab0 severstal
# launch evaluate --evaluator predict --job-config c041f9d41605254f805232999f143ab0 severstal
# launch evaluate --evaluator layer-output --job-config c041f9d41605254f805232999f143ab0 severstal
# launch collect --collector metric --job-config c041f9d41605254f805232999f143ab0 severstal
# launch collect --collector layer-output --job-config c041f9d41605254f805232999f143ab0 severstal
# launch collect --collector predict --job-config c041f9d41605254f805232999f143ab0 severstal
# launch collect --collector instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal
# launch visualize --visualizer metric --job-config c041f9d41605254f805232999f143ab0 severstal
# launch visualize --visualizer auc --job-config c041f9d41605254f805232999f143ab0 severstal
# launch visualize --visualizer predict --job-config c041f9d41605254f805232999f143ab0 severstal
# launch visualize --visualizer instance-metrics --job-config c041f9d41605254f805232999f143ab0 severstal
# launch visualize --visualizer layer-output --job-config c041f9d41605254f805232999f143ab0 severstal

# Weak Learner
# launch evaluate --evaluator metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch evaluate --evaluator predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch evaluate --evaluator layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch collect --collector metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch collect --collector layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch collect --collector predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch collect --collector instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch visualize --visualizer metric --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch visualize --visualizer auc --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch visualize --visualizer predict --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch visualize --visualizer instance-metrics --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
# launch visualize --visualizer layer-output --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal

# Strong Learner
# launch collect --collector weight --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 1
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --aggregators average_before
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --aggregators average_after
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --aggregators noisy_or
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --aggregators vote
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --aggregators average_before
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --aggregators average_after
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --aggregators noisy_or
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --aggregators vote
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold0
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold1
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold2
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold3
# **CHECK THIS ONE** sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold4
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold5
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold6
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold7
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold8
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds fold9

 # launch collect --collector weight --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 2
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --aggregators average_before
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --aggregators average_after
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --aggregators noisy_or
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --aggregators vote
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --aggregators average_before
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --aggregators average_after
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --aggregators noisy_or
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --aggregators vote
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold0
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold1
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold2
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold3
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold4
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold5
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold6
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold7
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold8
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds fold9

# launch collect --collector weight --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 3
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --aggregators average_before
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --aggregators average_after
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --aggregators noisy_or
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --aggregators vote
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --aggregators average_before
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --aggregators average_after
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --aggregators noisy_or
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --aggregators vote
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold0
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold1
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold2
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold3
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold4
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold5
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold6
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold7
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold8
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds fold9

# launch collect --collector weight --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal --classes 4
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --aggregators average_before
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --aggregators average_after
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --aggregators noisy_or
# sbatch run.sh evaluate severstal --evaluator metric --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --aggregators vote
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --aggregators average_before
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --aggregators average_after
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --aggregators noisy_or
# sbatch run.sh evaluate severstal --evaluator predict --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --aggregators vote
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold0
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold1
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold2
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold3
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold4
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold5
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold6
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold7
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold8
# sbatch run.sh evaluate severstal --evaluator layer-output --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds fold9


# Boosted Weak Learner
# launch evaluate --evaluator metric --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch evaluate --evaluator predict --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch evaluate --evaluator layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch collect --collector metric --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch collect --collector layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch collect --collector predict --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch collect --collector instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch collect --collector train --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch visualize --visualizer metric --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch visualize --visualizer auc --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch visualize --visualizer predict --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch visualize --visualizer instance-metrics --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch visualize --visualizer layer-output --job-config c4fc907d76c4adc975169e34d32b95df severstal
# launch visualize --visualizer boost --job-config c4fc907d76c4adc975169e34d32b95df severstal
