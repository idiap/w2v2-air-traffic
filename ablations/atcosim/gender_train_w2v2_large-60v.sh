#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# Base script to fine-tunine a wav2vec 2.0 large-960h-lv60-self model with ATCOSIM database
# We call the fine-tuning BASH script: src/run_asr_fine_tuning.sh
# which later calls PYTHON script: src/run_speech_recognition_ctc.py

#######################################
# COMMAND LINE OPTIONS,
# You can pass a qsub command (SunGrid Engine)
#       :an example is passing --cmd "src/sge/queue.pl h='*'-V", add your configuration
cmd="none"

# train/test subsets
dataset_name="experiments/data/atcosim_corpus/train"
eval_dataset_name="experiments/data/atcosim_corpus/test"

# some static vars for this experiment
exp="experiments/results/gender_exp"
max_steps=5000
per_device_train_batch_size=16
gradient_acc=2
learning_rate="5e-4"
mask_time_prob="0.01"

# model for ablation:
model="facebook/wav2vec2-large-960h-lv60-self"

# define the mask_time_probs to test,
genders="male female"
genders=($genders)

# get number of runs depending on the learning rates to train
num_runs=${#genders[@]}

# data folder where to store the results
rm -rf $exp/.error; mkdir -p $exp; 

for ((i = 0; i < $num_runs; i++)); do
  (
    # get the output folder where to store the fine-tuned model
    gender_folder=${genders[i]}
    out_folder=${exp}/$gender_folder

    # calling the bash script
    bash src/run_asr_fine_tuning.sh \
      --cmd "$cmd" \
      --model-name-or-path "$model" \
      --dataset-name "${dataset_name}_${gender_folder}" \
      --eval-dataset-name "${eval_dataset_name}_${gender_folder}" \
      --max-steps "$max_steps" \
      --per-device-train-batch-size "$per_device_train_batch_size" \
      --gradient-acc "$gradient_acc" \
      --learning_rate "$learning_rate" \
      --mask-time-prob "$mask_time_prob" \
      --overwrite-dir "true" \
      --exp "$out_folder"

  ) || touch ${exp}/.error &
done
wait
if [ -f ${exp}/.error ]; then
  echo "$0: something went wrong while fine-tuning gender-based models on ATCOSIM"
  exit 1
fi

echo "Done training the Gender-based experiments with wav2vec2-large-960h-lv60-self ATCOSIM in: $exp"
exit 1
