# BREAST CANCER DETECTION IN CORE-NEEDLE BIOPSIES WITH NEURAL NETWORKS
### Glejdis Shkembi

This repo is the official implementation of my master thesis on "Breast cancer detection in core-needle biopsies with neural networks". 

## Abstract

This project explores a new deep learning method to quantify clinical information from breast cancer core-needle biopsy histopathological image data. We  extend further on the attention-based multiple insatnce learning classification pipeline of [Xu et al. 2021](https://arxiv.org/abs/2112.02222).  Breast cancer has become the greatest threat to women's health worldwide. Clinically, identification of axillary lymph node (ALN) metastasis and other tumor clinical characteristics such as ER, PR, and so on, are important for evaluating the prognosis and guiding the treatment for breast cancer patients.
        Here, we will make use of a publicly available dataset from 1058 patients that include annotations from two independent and experienced pathologists, which allows estimating aleatoric and epistemic uncertainties of the underlying data and newly developed machine learning models. 
        Different baseline state-of-the-art deep learning models from the literature were developed to estimate the metastatic status of ALN, and subsequently, an extensive ablation study on different data augmentation techniques, including basic and advanced methods, was performed. Lastly, the tumor extraction and expert annotations were removed from the classification pipeline and the model performance was analyzed. The developed models were compared to the baseline by means of AUC, accuracy, F1-measure, F2-measure, sensitivity, specificity, PPV and NPV for diagnostic predictions.  
        
## Setup

### Clone this repo

```bash
git clone https://github.com/glejdis/master_thesis.git
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
The dataset, named `original_WSI_patches`, to run the experiment without the tumor segmentation and annotaion step, can be found in the directory `/vol/datasets/BCNB_original` of the workstation and should be coppied/moved to `code/dataset`.

Notice: the `my_json` folder is missing a file named `train.json`, due to the large size it could not be pushed in this repository. The file can be found in the directory `/vol/datasets/BCNB_original/my_json` of the workstation and should be coppied/moved to `code/dataset/my_json`.

## Statistical Analysis

To perform the statistcal analysis and fit the logistic regression model on the clinical dataset, run the jupyter notebooks from folder `statistical analysis` in the following order: 

1. Clinical_data_preparation
2. Logistic_regression
3. Stat_analysis

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

To run any experiment of the DLCNB with the clinical data without the segmentation step, you can do as this:

```bash
cd code
bash run_NOS.sh ${experiment_index}
```

To run any experiment of the DLCNB with the clinical data and data augmentation strategies, you can do as this:

```bash
cd code
bash run_further_data_aug.sh ${experiment_index}
```

Furthermore, if you want to try other settings, please see `train.py` for more details.

Some of the best results obtained from our experiemnts are given in folders `plots`, `plots_no_clinical`, `plots_no_Segmentation` and `logs`.
