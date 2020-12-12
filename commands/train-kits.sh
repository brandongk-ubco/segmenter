#!/bin/bash

find ./jobs -name '*.env' -exec sh -c 'export $(cat {} | xargs) && launch configure kits19' \;
find "$OUTPUT_DIR/severstal" -type d -printf '%f\n' | xargs -n1 -I % sh -c '{ sbatch run.sh train severstal --job-config % --classes 3 --folds 0; sleep 5; }'

find . -maxdepth 1 -type d | xargs -P 5 -I % sh -c 'rsync -a --bwlimit=50000 $(echo bgk@cedar.computecanada.ca:/scratch/bgk/results/seversal/%/ %/)'


rsync -avzm --stats --safe-links --ignore-existing --dry-run --human-readable bgk@cedar.computecanada.ca:/scratch/bgk/results/ . > /tmp/transfer.log

cat /tmp/transfer.log | parallel --will-cite -j 5 rsync -avzm --relative --stats --safe-links --ignore-existing --human-readable bgk@cedar.computecanada.ca:/scratch/bgk/results/{} . > result.log

# launch is-complete --job-config 89489ee3b0c0504c424df68d1672f0cf kits19
# launch is-complete --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19
# launch is-complete --job-config 54e053312ee47fe203400e055fa93be8 kits19
launch is-complete --job-config 89030ed7105f2d5c95654f5429366a55 kits19

#BASELINE
sbatch run.sh train kits19 --job-config 89489ee3b0c0504c424df68d1672f0cf --classes kidney
sbatch run.sh train kits19 --job-config 89489ee3b0c0504c424df68d1672f0cf --classes tumor

#UNBOOSTED
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 0
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 1
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 2
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 3
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 4
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 5
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 6
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 7
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 8
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds 9
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 0
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 1
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 2
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 3
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 4
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 5
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 6
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 7
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 8
# sbatch run.sh train kits19 --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds 9

#LARGE
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 0
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 1
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 2
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 3
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 4
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 5
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 6
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 7
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 8
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds 9
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 0
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 1
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 2
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 3
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 4
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 5
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 6
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 7
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 8
sbatch run.sh train kits19 --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds 9

#BOOSTED
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 0
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 1
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 2
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 3
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 4
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 5
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 6
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 7
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 8
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds 9
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 0
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 1
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 2
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 3
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 4
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 5
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 6
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 7
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 8
sbatch run.sh train kits19 --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds 9
