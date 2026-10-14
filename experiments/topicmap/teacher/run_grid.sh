#!/usr/bin/env bash
# The learning curve, then the staircase (teacher -> big student -> pool -> small students). Results: curve_results.jsonl
set -u
cd ~/Projects/tiny-classifiers
E=~/Projects/scholarsreadinglist/search-dev/experiments/topicmap/teacher
export CUDA_VISIBLE_DEVICES=1 TRITON_LIBCUDA_PATH=/run/opengl-driver/lib TOKENIZERS_PARALLELISM=false
run() { uv run -q --with scikit-learn python $E/curve.py "$@" 2>&1 | grep '^{' ; }
NS="250 500 1000 2000 4000 8000 17580"
for s in 0 1 2; do for n in $NS; do run --model jhu-clsp/ettin-encoder-17m --n $n --seed $s; done; done
for n in $NS; do run --model jhu-clsp/ettin-encoder-68m --n $n --seed 0; done
for n in $NS; do
  case $n in 1000|4000|17580) run --model jhu-clsp/ettin-encoder-150m --n $n --seed 0 --predict pool ;;
             *) run --model jhu-clsp/ettin-encoder-150m --n $n --seed 0 ;; esac
done
# staircase: the 150M student's labels on the 135k pool + the same N teacher labels -> a 17M student
for n in 1000 4000 17580; do
  run --model jhu-clsp/ettin-encoder-17m --n $n --seed 0 --pseudo $E/preds/pool__ettin-encoder-150m__n${n}__s0.json --epochs 2 --tag staircase
done
echo GRID DONE
