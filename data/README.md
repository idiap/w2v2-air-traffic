# Data preparation scripts 

## Table of Contents
- [Downloading Data](#downloading-data)
- [Preparing the Data](#preparing-the-data)
- [Cite us](#how-to-cite-us)

---
## Downloading Data

For the experiments carried out in this paper, you need to download the data folder one by one as follows:

### **ATCO2 test set corpus**

- Download (purchase) the full test set (used in the paper): http://catalog.elra.info/en-us/repository/browse/ELRA-S0484/
- **Download a free sample of the test set** (only contains 1 hour of data): https://www.atco2.org/data

### **UWB-ATCC corpus**

The Air Traffic Control Communication corpus or UWB-ATCC corpus, contains recordings of communication between air traffic controllers and pilots. The speech is manually transcribed and labeled with the information about the speaker (pilot/controller, not the full identity of the person). The audio data format is: 8kHz, 16bit PCM, mono.

**You can download this corpus for free in:** https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0

This item is Publicly Available and licensed under:
Attribution-NonCommercial-NoDerivs 3.0 Unported (CC BY-NC-ND 3.0)

### **ATCOSIM corpus**

The ATCOSIM Air Traffic Control Simulation Speech corpus is a speech database of air traffic control (ATC) operator speech, provided by Graz University of Technology (TUG) and Eurocontrol Experimental Centre (EEC). It consists of 10 hrs of speech data, which were recorded during ATC real-time simulations using a close-talk headset microphone. The utterances are in English language and pronounced by ten non-native speakers. 

**You can download this corpus for free in:** https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html

### **LDC-ATCC corpus**

The Air Traffic Control Complete (LDC94S14A) or LDC-ATCC corpus is comprised of recorded speech for use in supporting research and development activities in the area of robust speech recognition in domains similar to air traffic control (several speakers, noisy channels, relatively small vocabulary, constrained languaged, etc.) The audio data is composed of voice communication traffic between various controllers and pilots. The audio files are 8 KHz, 16-bit linear sampled data, representing continuous monitoring, without squelch or silence elimination, of a single FAA frequency for one to two hours.

You can purchase it and download here: https://catalog.ldc.upenn.edu/LDC94S14A


---
## Preparing the Data

The folder containing the preparation scripts to format and prepare each dataset looks like:

```
data/databases/
├── atco2_corpus
│   └── data_prepare_atco2_corpus.sh
├── atcosim_corpus
│   ├── data_prepare_atcosim_corpus.sh
│   └── link_acronyms.sh
├── ldc_atcc
│   ├── data_prepare_ldc_atcc_corpus.sh
│   ├── link_acronyms.sh
│   └── parse_lisp_array.sh
└── uwb_atcc
    ├── data_prepare_uwb_atcc_corpus.sh
    └── spk_id_tagger.py
```

You can format only one database and do all the experiments with it! **For instance**, [UWB-ATCC corpus](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0) and [ATCOSIM corpus](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html) are completly free to download and use! You can start right away with it!

### Prepare only one database: UWB-ATCC use case

For preparing one database, you can simply go to root directory and run:

```bash 
conda activate w2v2_asr
bash data/databases/data_prepare_uwb_atcc.sh
```

That will generate the files required for all the experiments in `experiments/data/uwb_atcc`. 


### Prepare all 4 databases

Don't worry! We have prepared a bash script/wrapper to format and prepare the 4 databases simultaneously. Go to the project root directory (one leve up) and run:

```bash 
conda activate w2v2_asr
bash data/PREPARE_AND_FORMAT_DATASETS.sh
```

This will prepare each dataset in KALDI format (**you DO NOT need to install KALDI**). The outputs are generated in the `experiments/data/*` folder, and its structure should be:

```
experiments/data/
├── atco2_corpus
│   └── prep
├── atcosim_corpus
│   ├── prep
│   ├── test
│   ├── test_female
│   ├── test_male
│   ├── train
│   ├── train_female
│   └── train_male
├── ldc_atcc
│   ├── prep
│   ├── test
│   └── train
└── uwb_atcc
    ├── prep
    ├── test
    └── train
```

Where `atco2_corpus`, `ldc_atcc`, `uwb_atcc` and `atcosim` are the 4 **public datasets** used in the [our Wav2Vec 2.0 implementation](https://arxiv.org/abs/2203.16822).

Each folder contains a `train` and `text` folder. Each folder contains several files. However, the most important ones are: 


- text: contains the transcripts, format: `utt_id transcript`
- segments:contains the acoutisc segmentation information, format `utt_id recording_id t_begin t_end`
- wav.scp: contains the path to the recordings (wav/sph/etc), format: `recording_id /path/to/wav/`


```
!!! LDC-ATC, UWB-ATCC and ATCOSIM corpora are used for training and testing, while ATCO2-test-set corpus only for testing. 
```

---
# How to cite us

If you use this code for your research, please cite our paper with:

Zuluaga-Gomez, J., Prasad, A., Nigmatulina, I., Sarfjoo, S., Motlicek, P., Kleinert, M., ... & Zhan, Q. (2022). How Does Pre-trained Wav2Vec2. 0 Perform on Domain Shifted ASR? An Extensive Benchmark on Air Traffic Control Communications. 2022 IEEE Spoken Language Technology Workshop (SLT), Doha, Qatar.

or use the bibtex item:

```
@article{zuluaga2022how,
    title={How Does Pre-trained Wav2Vec2. 0 Perform on Domain Shifted ASR? An Extensive Benchmark on Air Traffic Control Communications},
    author={Zuluaga-Gomez, Juan and Prasad, Amrutha and Nigmatulina, Iuliia and Sarfjoo, Saeed and Motlicek, Petr and Kleinert, Matthias and Helmke, Hartmut and Ohneiser, Oliver and Zhan, Qingran},
    journal={IEEE Spoken Language Technology Workshop (SLT), Doha, Qatar},
    year={2022}
  }
```
