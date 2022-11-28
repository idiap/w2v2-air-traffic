#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# Base script to upload a Wav2Vec 2.0 model to HuggingFace hub
# This model has been fine-tuned on ATC data
#######################################
# COMMAND LINE OPTIONS,
set -euo pipefail

path_to_lm="experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary"
path_to_model="experiments/results/robust_model/wav2vec2-xls-r-300m/atcosim_corpus/0.0ld_0.05ad_0.05attd_0.0fpd_0.03mtp_10mtl_0.0mfp_10mfl/"
output_folder="notebooks"
repository_name="wav2vec2-xls-r-300m-en-atc-atcosim"

. data/utils/parse_options.sh

output_folder=$output_folder/$repository_name
mkdir -p $output_folder

echo "*** Uploading model to HuggingFace hub ***"
echo "*** repository will be stored in: $output_folder ***"

python3 notebooks/upload_model_with_lm.py \
  --lm "$path_to_lm" \
  --w2v2 "$path_to_model" \
  --output-folder "$output_folder" \
  --output-repo-name "$repository_name"

echo "Done uploading the model in ${path_to_model} with LM"
exit 0
