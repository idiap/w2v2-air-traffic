#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# Base script to evaluate a Wav2Vec 2.0 model with PyCTCdecode and Transformers
#######################################
# COMMAND LINE OPTIONS,
set -euo pipefail

path_to_lm="experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary"
path_to_model="experiments/results/baselines/wav2vec2-large-960h-lv60-self/uwb_atcc/0.0ld_0.05ad_0.05attd_0.0fpd_0.075mtp_12mtl_0.075mfp_12mfl/checkpoint-10000"
test_set="experiments/data/uwb_atcc/test"

print_output="true"

. data/utils/parse_options.sh

echo "*** About to evaluate a Wav2Vec 2.0 model***"
echo "*** Dataset in: $test_set ***"
echo "*** Output folder: $(dirname $path_to_model)/output ***"

python3 src/eval_model.py \
  --language-model "$path_to_lm" \
  --pretrained-model "$path_to_model" \
  --print-output "$print_output" \
  --test-set "$test_set"

echo "Done evaluating model in ${path_to_model} with LM"
exit 0
