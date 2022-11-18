#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# Base script to fine-tunine a wav2vec 2.0 model with LDC-ATCC database
# We call the fine-tuning BASH script: src/run_asr_fine_tuning.sh
# which later calls PYTHON script: src/run_speech_recognition_ctc.py

#######################################
# COMMAND LINE OPTIONS,
# You can pass a qsub command (SunGrid Engine)
#       :an example is passing --cmd "src/sge/queue.pl h='*'-V", add your configuration
cmd="none"

model_name_or_path="facebook/wav2vec2-base"
dataset_name="experiments/data/ldc_atcc/train"
eval_dataset_name="experiments/data/ldc_atcc/test"
exp="experiments/results/baselines"

# calling the bash script
bash src/run_asr_fine_tuning.sh \
  --cmd "$cmd" \
  --model-name-or-path "$model_name_or_path" \
  --dataset-name "$dataset_name" \
  --eval-dataset-name "$eval_dataset_name" \
  --max-steps "10000" \
  --per-device-train-batch-size "24" \
  --gradient-acc "3" \
  --overwrite-dir "true" \
  --exp "$exp"

echo "Done training of baseline model for LDC-ATCC database"
exit 0
