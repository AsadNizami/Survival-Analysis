# Benchmarking Vision Encoders for Survival Analysis using Histopathological Images

## Abstract
Cancer is a complex disease characterized by the uncontrolled growth of abnormal cells in the body but can be prevented and even cured when detected early. Advanced medical imaging has introduced Whole Slide Images (WSIs). When combined with deep learning techniques, it can be used to extract meaningful features. These features are useful for various tasks such as classification and segmentation. There have been numerous studies involving the use of WSIs for survival analysis. Hence, it is crucial to determine their effectiveness for specific use cases. In this paper, we compared three publicly available vision encoders- UNI, Phikon and ResNet18 which are trained on millions of histopathological images, to generate feature embedding for survival analysis. WSIs cannot be fed directly to a network due to their size. We have divided them into 256 $\times$ 256 pixels patches and used a vision encoder to get feature embeddings. These embeddings were passed into an aggregator function to get representation at the WSI level which was then passed to a Long Short Term Memory (LSTM) based risk prediction head for survival analysis. Using breast cancer data from The Cancer Genome Atlas Program (TCGA) and k-fold cross-validation, we demonstrated that transformer-based models are more effective in survival analysis and achieved better C-index on average than ResNet-based architecture.

![experimental_setup](https://github.com/user-attachments/assets/b44eeba4-e8bd-453c-aae8-db38395d6bc1)

*Fig 1: Medical Image Analysis Pipeline for Risk Prediction*

## Training and Evaluation
```
git clone https://github.com/AsadNizami/Y-Net.git
cd Y-Net/src
```
Configure the `config.py` file for your environment

```
python train.py
```

## Results
### Average Validation C-index and Standard Deviation

| Model                  | C-index               |
|------------------------|-----------------------|
| UNI                    | 0.5842 (± 0.0366)     |
| Phikon                 | 0.5872 (± 0.1073)     |
| ResNet18-TIAToolbox    | 0.5689 (± 0.0139)     |
