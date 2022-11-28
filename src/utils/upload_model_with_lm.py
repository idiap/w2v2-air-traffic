#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

DESCRIPTION="Script to upload model to HuggingFace hub"

import argparse
import os
import sys
import subprocess
from pathlib import Path

from pyctcdecode import build_ctcdecoder
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ProcessorWithLM,
)
import huggingface_hub
from huggingface_hub import HfApi, create_repo


def get_kenlm_processor(model_path, path_lm=None):
    """Function that instantiate the models and then gives them back for evaluation"""

    path_tokenizer = os.path.dirname(model_path)

    # check that we send the right dir where the model is stored
    if Path(model_path).is_dir():
        processor = AutoProcessor.from_pretrained(path_tokenizer)
        model = AutoModelForCTC.from_pretrained(model_path)
    else:
        print(f"Error. Models were not found in {model_path}")

    # In case we don't pass any language model path, we just send back the model and processor
    if path_lm is None:
        return processor, None, None, model

    vocab = processor.tokenizer.convert_ids_to_tokens(
        range(0, processor.tokenizer.vocab_size)
    )
    # we need to add these tokens in the tokenizer
    vocab.append("<s>")
    vocab.append("</s>")

    # instantiate the tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        path_tokenizer + "/vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    # load CTC decoder WITH a LM and HuggingFace processor with CTC decoder and LM
    ctcdecoder_kenlm = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=path_lm,
    )
    # HuggingFace processor with CTC decoder (with LM)
    processor_ctc_kenlm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=tokenizer,
        decoder=ctcdecoder_kenlm,
    )

    return processor, processor_ctc_kenlm, model


def parse_args():
    """parser"""
    parser = argparse.ArgumentParser(description=DESCRIPTION,
        usage="Usage: upload_model_with_lm.py --ph True --lm /path/to/lm_4g._binary --w2v2 /path/to/model/checkpoint"
    )

    # optional,
    parser.add_argument(
        "--lm",
        "--language-model",
        dest="path_lm",
        default=None,
        help="Directory with an in-domain LM. Needs to match the symbol table",
    )

    # must give,
    parser.add_argument(
        "--output-repo-name",
        dest="output_repo_name",
        required=True,
        help="name of the output repository name, e.g., wav2vec2-xls-r-300m-en-atc-atcosim",
    )
    parser.add_argument(
        "--w2v2",
        "--pretrained-model",
        dest="path_model",
        required=True,
        help="Directory with pre-trained Wav2Vec 2.0 model (or XLS-R-300m).",
    )
    parser.add_argument(
        "--of",
        "--output-folder",
        dest="output_folder",
        required=True,
        help="Directory where to put the model with its model-card.",
    )

    return parser.parse_args()

def main():
    """Main code execution"""
    args = parse_args()

    path_model = args.path_model + '/'
    path_lm = args.path_lm
    path_output_dir = args.output_folder

    # get the repo_id name, based on <args>,
    output_repo_name = args.output_repo_name
    get_user_name = huggingface_hub.whoami()['name']
    repo_id = f"{get_user_name}/{output_repo_name}"

    if not Path(path_lm).is_file():
        print(f"You pass a path to LM ({path_lm}), but file does not exists. Exit")
        sys.exit(1)
    else:
        print("Integrating a LM by shallow fusion, results should be better")
    
    # create the output folder if is not present
    if len(os.listdir(path_output_dir)) > 0:
        print(f"you passed a non-empty folder ({path_output_dir}), this cannot be managed automatically by HF.")
        sys.exit(1)

    # First, create the repo in HuggingFace-Hub, we fetch the local username 
    try:
        create_repo(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"Repo already create, the error was: \n{e}")
        print(f"\n we continue...")

    print("*** copying model to output folder, loading... ***")    
    if Path(path_output_dir + '/pytorch_model.bin').is_file():
        print("pytorch_model.bin is already present in the output folder, not copying it")
        print(f"output folder is: {path_output_dir}")
    else:
        print(f"Copying the model folder to: {path_output_dir}")
        # copy the folder with the model and log subfolder
        subprocess.run(
            f"cp {path_model}/* {path_output_dir}/", 
            shell=True, 
            stderr=sys.stderr, 
            stdout=sys.stdout
        )
        subprocess.run(
            f"cp -r {path_model}/log {path_output_dir}/", 
            shell=True, 
            stderr=sys.stderr,
            stdout=sys.stdout
        )
    
    print("*** Loading the Wav2Vec 2.0 model, loading... ***")
    # We also load the models with CTC decoder with LM
    # import ipdb; ipdb.set_trace()
    processor, processor_ctc_kenlm, model = get_kenlm_processor(path_model, path_lm)
    processor_ctc_kenlm.save_pretrained(path_output_dir)
    
    print("*** Uploading the model to HuggingFace Hub... ***")
    api = HfApi()
    api.upload_folder(
        folder_path=path_output_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns="**/checkpoint-*",
        commit_message="updating the repo with the fine-tuned model"
    )

    return None


if __name__ == "__main__":
    main()
