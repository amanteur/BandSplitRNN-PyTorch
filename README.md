# BandSplitRNN Pytorch
An unofficial PyTorch implementation of the paper [Music Source Separation with Band-split RNN](https://arxiv.org/pdf/2209.15174.pdf).

---
# Dependecies

Python version - **3.10**.  
To install dependencies please run:
```
pip install -r requirements.txt
```
Additionally, **ffmpeg** should be installed in the venv.
```
conda install -c conda-forge ffmpeg
```
---
# TODOs
1. Add augmentations
2. Train
3. Delete non-useful gradnorms in logger 
4. Add repo structure in README.md
---
To cite the paper, please use:
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