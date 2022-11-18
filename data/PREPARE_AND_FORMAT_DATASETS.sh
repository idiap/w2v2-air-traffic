#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# This script is intended to format all the databases for the text-based diarization
# paper

# Each database has a preparation script, you can follow each of them 
# in their respective folders

# Define the paths were you downloaded the data here:
LDC_ATCC_PATH="/usr/downloads/LDC_ATCC/atc0_comp/*/data"
UWB_ATCC_PATH="/usr/downloads/PILSEN_DB/ZCU_CZ_ATC"
ATCO2_CORPUS_PATH="/usr/downloads/ATCO2-ASRdataset-v1_final/DATA"
ATCOSIM_PATH="/usr/downloads/atcosim"

# Folders where to store the formatted data
EXP_FOLDER=experiments/data/

############################################################
# Preparing LDC-ATCC corpus 
echo "**** Preparing LDC-ATCC corpus ****"

bash data/databases/ldc_atcc/data_prepare_ldc_atcc_corpus.sh \
  --EXP-FOLDER $EXP_FOLDER \
  --DATA "$LDC_ATCC_PATH"

############################################################
# Preparing UWB-ATCC corpus 
echo "**** Preparing UWB-ATCC corpus ****"

bash data/databases/uwb_atcc/data_prepare_uwb_atcc_corpus.sh \
  --EXP-FOLDER $EXP_FOLDER \  
  --DATA "$UWB_ATCC_PATH"

############################################################
# Preparing ATCO2 corpus 
echo "**** Preparing ATCO2-test-set corpus ****"

bash data/databases/atco2_corpus/data_prepare_atco2_corpus.sh \
  --EXP-FOLDER $EXP_FOLDER \
  --DATA "$ATCO2_CORPUS_PATH"

############################################################
# Preparing ATCOSIM corpus 
echo "**** Preparing ATCOSIM corpus ****"

bash data/databases/atco2_corpus/data_prepare_atcosim_corpus.sh \
  --EXP-FOLDER $EXP_FOLDER \  
  --DATA "$ATCOSIM_PATH"

echo "**** Datasets were sucessfully created ****"
exit 0
