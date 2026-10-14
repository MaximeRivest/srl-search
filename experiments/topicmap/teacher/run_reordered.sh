#!/usr/bin/env bash
# Let the 150M n=250 point finish, stop the grid, run the 0.8B few-shot tests, then the rest of the grid.
set -u
E=~/Projects/scholarsreadinglist/search-dev/experiments/topicmap/teacher
until grep -q '"ettin-encoder-150m", "n": 250,' $E/curve_results.jsonl; do sleep 20; done
systemctl --user stop srl-curve
cd ~/Projects/tiny-classifiers
export CUDA_VISIBLE_DEVICES=1 TRITON_LIBCUDA_PATH=/run/opengl-driver/lib TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
few() { uv run -q --with peft --with scikit-learn python $E/fewshot_llm.py "$@" 2>&1 | grep '^{' ; }
few --n 250 --steps 150
few --n 250 --steps 150 --bidirectional
few --n 1000 --steps 300
few --n 1000 --steps 300 --bidirectional
echo FEWSHOT DONE
run() { uv run -q --with scikit-learn python $E/curve.py "$@" 2>&1 | grep '^{' ; }
for n in 500 1000 2000 4000 8000 17580; do
  case $n in 1000|4000|17580) run --model jhu-clsp/ettin-encoder-150m --n $n --seed 0 --predict pool ;;
             *) run --model jhu-clsp/ettin-encoder-150m --n $n --seed 0 ;; esac
done
for n in 1000 4000 17580; do
  run --model jhu-clsp/ettin-encoder-17m --n $n --seed 0 --pseudo $E/preds/pool__ettin-encoder-150m__n${n}__s0.json --epochs 2 --tag staircase
done
echo GRID DONE
