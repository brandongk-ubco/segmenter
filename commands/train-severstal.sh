#!/usr/bin/env bash

set -euo pipefail
 
find "$OUTPUT_DIR/severstal" -type d -printf '%f\n' | xargs -n1 -I % sh -c '{ echo sbatch run.sh train severstal --job-config % --classes 4 --folds 0; sleep 5; }'

find ./jobs -name '*.env' -exec sh -c 'export $(cat {} | xargs) && launch configure severstal' \;

#launch is-complete --job-config c041f9d41605254f805232999f143ab0 severstal
#launch is-complete --job-config 8ba8ae882c396d08aaa3332167cd7aeb severstal
#launch is-complete --job-config c4fc907d76c4adc975169e34d32b95df severstal
launch is-complete --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 severstal

# BASELINE
#sbatch run.sh train severstal --job-config c041f9d41605254f805232999f143ab0 --classes 1
#sbatch run.sh train severstal --job-config c041f9d41605254f805232999f143ab0 --classes 2
#sbatch run.sh train severstal --job-config c041f9d41605254f805232999f143ab0 --classes 3
#sbatch run.sh train severstal --job-config c041f9d41605254f805232999f143ab0 --classes 4

# UNBOOSTED
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 0
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 1
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 2
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 3
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 4
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 5
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 6
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 7
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 8
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 1 --folds 9
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 0
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 1
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 2
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 3
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 4
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 5
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 6
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 7
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 8
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 2 --folds 9
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 0
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 1
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 2
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 3
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 4
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 5
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 6
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 7
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 8
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 3 --folds 9
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 0
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 1
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 2
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 3
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 4
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 5
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 6
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 7
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 8
#sbatch run.sh train severstal --job-config 8ba8ae882c396d08aaa3332167cd7aeb --classes 4 --folds 9

# LARGE
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 0
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 1
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 2
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 3
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 4
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 5
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 6
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 7
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 8
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 1 --folds 9
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 0
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 1
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 2
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 3
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 4
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 5
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 6
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 7
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 8
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 2 --folds 9
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 0
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 1
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 2
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 3
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 4
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 5
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 6
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 7
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 8
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 3 --folds 9
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 0
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 1
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 2
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 3
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 4
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 5
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 6
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 7
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 8
#sbatch run.sh train severstal --job-config 4e4c5e07f5728b9b5d7ec365b8e211c8 --classes 4 --folds 9

# BOOSTED
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 0
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 1
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 2
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 3
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 4
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 5
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 6
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 7
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 8
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 1 --folds 9
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 0
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 1
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 2
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 3
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 4
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 5
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 6
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 7
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 8
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 2 --folds 9
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 0
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 1
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 2
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 3
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 4
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 5
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 6
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 7
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 8
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 3 --folds 9
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 0
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 1
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 2
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 3
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 4
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 5
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 6
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 7
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 8
#sbatch run.sh train severstal --job-config c4fc907d76c4adc975169e34d32b95df --classes 4 --folds 9
