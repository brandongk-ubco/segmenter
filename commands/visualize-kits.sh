#!/usr/bin/env bash

conda activate segmenter

set -euo pipefail

# Baseline
launch collect --collector metric --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch collect --collector predict --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch collect --collector instance-metrics --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch visualize --visualizer metric --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch visualize --visualizer auc --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch visualize --visualizer predict --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch visualize --visualizer instance-metrics --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch collect --collector layer-output --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch visualize --visualizer layer-output --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch visualize --visualizer best-threshold --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch visualize --visualizer best-threshold --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor
launch collect --collector confusion --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes tumor

launch collect --collector metric --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch collect --collector predict --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch collect --collector instance-metrics --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch visualize --visualizer metric --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch visualize --visualizer auc --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch visualize --visualizer predict --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch visualize --visualizer instance-metrics --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch collect --collector layer-output --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch visualize --visualizer layer-output --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch visualize --visualizer best-threshold --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney
launch collect --collector confusion --job-config 89489ee3b0c0504c424df68d1672f0cf kits19 --classes kidney

launch visualize --visualizer confusion --job-config 89489ee3b0c0504c424df68d1672f0cf kits19

# Weak Learner
launch collect --collector metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch collect --collector predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch collect --collector instance-metrics --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch visualize --visualizer metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch visualize --visualizer auc --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch visualize --visualizer predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch visualize --visualizer instance-metrics --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch collect --collector layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch visualize --visualizer layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch visualize --visualizer best-threshold --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor
launch collect --collector confusion --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes tumor

launch collect --collector metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch collect --collector predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch collect --collector instance-metrics --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch visualize --visualizer metric --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch visualize --visualizer auc --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch visualize --visualizer predict --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch visualize --visualizer instance-metrics --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch collect --collector layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch visualize --visualizer layer-output --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch visualize --visualizer best-threshold --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney
launch collect --collector confusion --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19 --classes kidney

launch visualize --visualizer confusion --job-config f9e2afd0b43fb3f43ab7dc2a95bd6368 kits19

# Boosted Learner
launch collect --collector metric --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch collect --collector predict --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch collect --collector instance-metrics --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch collect --collector train --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch visualize --visualizer metric --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch visualize --visualizer auc --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch visualize --visualizer predict --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch visualize --visualizer instance-metrics --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch visualize --visualizer boost --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch collect --collector layer-output --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch visualize --visualizer layer-output --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch visualize --visualizer best-threshold --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor
launch collect --collector confusion --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes tumor

launch collect --collector layer-output --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch visualize --visualizer layer-output --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch collect --collector metric --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch collect --collector predict --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch collect --collector instance-metrics --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch collect --collector train --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch visualize --visualizer metric --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch visualize --visualizer auc --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch visualize --visualizer predict --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch visualize --visualizer instance-metrics --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch visualize --visualizer boost --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch visualize --visualizer best-threshold --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney
launch collect --collector confusion --job-config 54e053312ee47fe203400e055fa93be8 kits19 --classes kidney

launch visualize --visualizer confusion --job-config 54e053312ee47fe203400e055fa93be8 kits19

# Strong Learner
launch collect --collector metric --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch collect --collector predict --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch collect --collector instance-metrics --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch visualize --visualizer metric --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch visualize --visualizer auc --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch visualize --visualizer predict --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch visualize --visualizer instance-metrics --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch collect --collector layer-output --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch visualize --visualizer layer-output --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch visualize --visualizer best-threshold --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor
launch collect --collector confusion --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes tumor

launch collect --collector metric --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch collect --collector predict --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch collect --collector instance-metrics --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch visualize --visualizer metric --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch visualize --visualizer auc --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch visualize --visualizer predict --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch visualize --visualizer instance-metrics --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch collect --collector layer-output --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch visualize --visualizer layer-output --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch visualize --visualizer best-threshold --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney
launch collect --collector confusion --job-config 89030ed7105f2d5c95654f5429366a55 kits19 --classes kidney

launch visualize --visualizer confusion --job-config 89030ed7105f2d5c95654f5429366a55 kits19