#!/usr/bin/env bash
# After the 0.8B few-shot runs: the 0.8B (250 Kimi labels) labels 20k pool papers; Ettin-17M learns from them
# (with and without the 250 Kimi labels); then the rest of the 150M grid and its staircase.
set -u
E=~/Projects/scholarsreadinglist/search-dev/experiments/topicmap/teacher
until journalctl --user -u srl-reorder -o cat --no-pager | grep -q "FEWSHOT DONE"; do sleep 15; done
systemctl --user stop srl-reorder
cd ~/Projects/tiny-classifiers
export CUDA_VISIBLE_DEVICES=1 TRITON_LIBCUDA_PATH=/run/opengl-driver/lib TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
few() { uv run -q --with peft --with scikit-learn python $E/fewshot_llm.py "$@" 2>&1 | grep '^{' ; }
run() { uv run -q --with scikit-learn python $E/curve.py "$@" 2>&1 | grep '^{' ; }
few --n 250 --steps 150 --predict pool20k
P=$E/preds/pool20k__Qwen3.5-0.8B__n250__s0.json
run --model jhu-clsp/ettin-encoder-17m --n 250 --seed 0 --pseudo $P --epochs 3 --tag cheap-teacher-0.8B
run --model jhu-clsp/ettin-encoder-68m --n 250 --seed 0 --pseudo $P --epochs 3 --tag cheap-teacher-0.8B
echo CHEAP DONE
for n in 500 1000 2000 4000 8000 17580; do
  case $n in 1000|4000|17580) run --model jhu-clsp/ettin-encoder-150m --n $n --seed 0 --predict pool ;;
             *) run --model jhu-clsp/ettin-encoder-150m --n $n --seed 0 ;; esac
done
for n in 1000 4000 17580; do
  run --model jhu-clsp/ettin-encoder-17m --n $n --seed 0 --pseudo $E/preds/pool__ettin-encoder-150m__n${n}__s0.json --epochs 2 --tag staircase
done
echo GRID DONE
