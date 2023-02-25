# BandSplitRNN Pytorch

An unofficial PyTorch implementation of the paper [Music Source Separation with Band-split RNN](https://arxiv.org/pdf/2209.15174.pdf).

---
## Table of Contents
0. [TODOs](#todos)
1. [Dependencies](#dependencies)
2. [Fast Start](#faststart)
3. [Repository structure](#structure)
4. [Train you model](#trainmodel)
   1. [Dataset preprocessing](#preprocessing)
   2. [Training](#train)
   3. [Evaluation](#eval)
8. [Citing](#cite)



---
<a name="todos"/>

## TODOs

- Train
- Add test.py pipeline 
- Add inference.py pipeline

---
<a name="dependencies"/>

# Dependencies

Python version - **3.10**.  
To install dependencies, run:
```
pip install -r requirements.txt
```
Additionally, **ffmpeg** should be installed in the venv.  
If using ``conda``, you can run:
```
conda install -c conda-forge ffmpeg
```

---
<a name="faststart"/>

## Fast Start

bla-bla

---
<a name="structure"/>

## Repository structure
The structure of this repository is as following:
```
├── src
│   ├── conf                        - hydra configuration files
│   │   └── **/*.yaml               
│   ├── data                        - directory with data processing modules
│   │   ├── __init__.py             
│   │   ├── augmentations.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── utils.py
│   ├── files                       - output files from prepare_dataset.py script
│   │   └── *.txt
│   ├── model                       - directory with neural networks modules 
│   │   ├── modules
│   │   │   ├── __init__.py
│   │   │   ├── bandsequence.py
│   │   │   ├── bandsplit.py
│   │   │   ├── maskestimation.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   └── bandsplitrnn.py
├── example                         - test example for inference.py
│   └── *.wav
├── .gitignore
├── README.md 
└── requirement.txt
```

---
<a name="trainmodel"/>

## Train your model

In this section model training pipeline is described.

<a name="preprocessing"/>

### Dataset preprocessing

Authors used MUSDB18-HQ dataset to train an initial source separation model.
You can access it via [zenodo](https://zenodo.org/record/3338373#.Y_jrMC96D5g).
After downloading, set path to this dataset as an environmental variable:
```
export MUSDB_DIR={MUSDB_DIR}
```

To speed up the training process, instead of loading whole files, 
we can precompute the indices of fragments we need to extract. 
To select these indices, the proposed Source Activity Detection algorithm was used.

The following script reads `musdb18` dataset and extracts salient fragments according to `target` source:
```
usage: prepare_dataset.py [-h] -i INPUT_DIR -o OUTPUT_DIR [--subset SUBSET] [--split SPLIT] [--sad_cfg SAD_CFG] [-t TARGET [TARGET ...]]

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Path to directory with musdb18 dataset
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to directory where output .txt file is saved
  --subset SUBSET       Train/test subset of dataset to process
  --split SPLIT         Train/valid split of train dataset. Used if subset=train
  --sad_cfg SAD_CFG     Path to Source Activity Detection config file
  -t TARGET [TARGET ...], --target TARGET [TARGET ...]
                        Target source. SAD will save salient fragments of vocal audio.

```
Output is saved to `{OUTPUT_DIR}/{TARGET}_{SUBSET}.txt` file. The structure of file is as following:
```
{MUSDB18 TRACKNAME}\t{START_INDEX}\t{END_INDEX}\n
```

---
<a name="train"/>

### Training

To train the model combination of `pytorch-lightning` and `hydra` was used.
All configuration files are stored in `src/conf` directory in `hydra`-friendly format.

To start training a model with given configurations, use the following script:
```
export CUDA_VISIBLE_DEVICES={DEVICE_NUM} && python train.py
```
To configure training process follow `hydra` [instructions](https://hydra.cc/docs/advanced/override_grammar/basic/).
By default, model is trained to extract `vocals`. To train a model to extract other sources, use the following scripts:
```
export CUDA_VISIBLE_DEVICES={DEVICE_NUM} && python train.py train_dataset.target=bass model=bandsplitrnnbass
export CUDA_VISIBLE_DEVICES={DEVICE_NUM} && python train.py train_dataset.target=drums model=bandsplitrnndrums
export CUDA_VISIBLE_DEVICES={DEVICE_NUM} && python train.py train_dataset.target=other
```

After training is started, the logging folder will be created for a particular experiment with a following path:
```
src/logs/bandsplitrnn/${now:%Y-%m-%d}_${now:%H-%M}/
```
This folder will have a following structure:
```
├── tb_logs
│   └── tensorboard_log_file    - main tensorboard log file 
├── weights
│   └── *.ckpt                  - lightning model checkpoint files.
└── hydra
    └──config.yaml             - hydra configuration and override files 
```

---
<a name="eval"/>

### Evaluation

bla-bla

---
<a name="cite"/>

## Citing

To cite this paper, please use:
```
@misc{https://doi.org/10.48550/arxiv.2209.15174,
  doi = {10.48550/ARXIV.2209.15174},
  url = {https://arxiv.org/abs/2209.15174},
  author = {Luo, Yi and Yu, Jianwei},
  keywords = {Audio and Speech Processing (eess.AS), Machine Learning (cs.LG), Sound (cs.SD), Signal Processing (eess.SP), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Music Source Separation with Band-split RNN},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```
