# BREAST CANCER DETECTION IN CORE-NEEDLE BIOPSIES WITH NEURAL NETWORKS

This repo is the official implementation of my paper "Breast cancer detection in core-needle biopsies with neural networks". 

## Abstract

Breast cancer (BC) has become the greatest threat to women's health worldwide. Clinically, identification of axillary lymph node (ALN) metastasis is important for evaluating prognosis and guiding  treatment. This paper aims at reproducing the results from \cite{Xu_2021}, and further extends their deep learning (DL) classification pipeline by quantifying clinical information from core-needle biopsy (CNB) images. We made use of a publicly available dataset of $1058$ patients. Different baseline state-of-the-art (SOTA) DL models were tested to estimate the metastatic status of ALNs. Subsequently, an extensive ablation study of different data augmentation techniques was performed. Lastly, tumor extraction and expert annotations were removed from the classification pipeline.  Our proposed model outperformed SOTA by $3.73$ $\%$. 
        
## Setup

### Clone this repo

```bash
git clone https://github.com/glejdis/CMT_code.git
```

### Environment

Create environment and install dependencies.

```bash
conda create -n DLCNBC python=3.9.12 -y
conda activate DLCNBC
pip install -r code/requirements.txt
```
        
 ### Dataset

For your convenience, we have provided preprocessed clinical data in `code/dataset`. The processed WSI patches can be downloaded from [here](https://drive.google.com/file/d/1wY5KIVixdwzZZq2m0IoqmBLp0YlwBAz6/view?usp=sharing) and unzip them by the following scripts:

```bash
cd code/dataset
# download paper_patches.zip
unzip paper_patches.zip
```

## Training

To train our models run the following:

> experiment_index:
> 
> 0. N0 vs N+(>0)
> 2. N0 vs N+(1-2) vs N+(>2)

To run any experiment of the DLCNB with the clinical data, you can do as this:

```bash
cd code
bash run.sh ${experiment_index}
```

To run any experiment of the DLCNB without the clinical data, you can do as this:

```bash
cd code
bash run_no_clinical.sh ${experiment_index}
```

To run any experiment of the DLCNB with the clinical data and data augmentation strategies, you can do as this:

```bash
cd code
bash run_further_data_aug.sh ${experiment_index}
```

Furthermore, if you want to try other settings, please see `train.py` for more details.
