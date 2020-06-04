#!/usr/bin/env bash

set -euo pipefail

# launch collect --collector weight --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89489ee3b0c0504c424df68d1672f0cf --classes kidney
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89489ee3b0c0504c424df68d1672f0cf --classes kidney
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89489ee3b0c0504c424df68d1672f0cf --classes kidney

# launch collect --collector weight --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89489ee3b0c0504c424df68d1672f0cf --classes tumor
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89489ee3b0c0504c424df68d1672f0cf --classes tumor
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89489ee3b0c0504c424df68d1672f0cf --classes tumor

#launch collect --collector weight --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
# sbatch run.sh evaluate kits19 --evaluator metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold5
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold6
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold7
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold8
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes kidney --folds fold9

#launch collect --collector weight --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
# sbatch run.sh evaluate kits19 --evaluator metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold5
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold6
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold7
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold8
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 --classes tumor --folds fold9

# launch collect --collector weight --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
# 1,246,010 parameters
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold5
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold6
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold7
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold8
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes kidney --folds fold9

# launch collect --collector weight --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold5
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold6
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold7
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold8
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 89030ed7105f2d5c95654f5429366a55 --classes tumor --folds fold9

# launch collect --collector weight --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold0b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold0b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold0b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold0b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold0b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold1b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold1b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold1b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold1b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold1b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold2b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold2b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold2b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold2b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold2b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold3b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold3b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold3b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold3b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold3b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold4b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold4b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold4b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold4b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold4b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold5b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold5b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold5b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold5b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold5b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold6b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold6b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold6b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold6b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold6b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold7b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold7b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold7b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold7b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold7b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold8b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold8b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold8b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold8b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold8b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold9b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold9b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold9b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold9b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes kidney --folds fold9b4

# launch collect --collector weight --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator metric --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --aggregators average_before
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --aggregators average_after
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --aggregators noisy_or
# sbatch run.sh evaluate kits19 --evaluator predict --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --aggregators vote
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold0b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold0b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold0b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold0b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold0b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold1b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold1b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold1b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold1b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold1b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold2b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold2b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold2b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold2b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold2b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold3b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold3b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold3b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold3b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold3b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold4b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold4b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold4b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold4b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold4b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold5b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold5b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold5b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold5b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold5b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold6b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold6b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold6b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold6b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold6b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold7b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold7b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold7b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold7b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold7b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold8b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold8b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold8b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold8b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold8b4
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold9b0
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold9b1
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold9b2
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold9b3
# sbatch run.sh evaluate kits19 --evaluator layer-output --job-config 54e053312ee47fe203400e055fa93be8 --classes tumor --folds fold9b4
