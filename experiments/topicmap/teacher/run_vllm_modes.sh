#!/usr/bin/env bash
# vLLM start-up and speed of the 0.8B decider: without compiled kernels (--eager), then compiled (cache warm).
cd ~/Projects/scholarsreadinglist/search-dev/experiments/topicmap/teacher
V=/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
for m in --eager ""; do
  echo "MODE ${m:-compiled}"
  t=$(date +%s)
  $V vllm_score.py ~/Projects/tiny-classifiers/models/frag-qwen08-n375 pool20k --mem 0.5 $m 2>&1 | grep -E '^\{|rror'
  echo "WALL $(( $(date +%s) - t )) s"
done
echo MODES DONE
