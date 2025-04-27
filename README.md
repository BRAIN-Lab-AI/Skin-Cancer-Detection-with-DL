# Skin Classification Classification Using Deep Learning Methods

## Introduction
Skin cancer classification is a critical application of deep learning in medical imaging, aiming to improve early diagnosis and treatment outcomes for malignant lesions such as melanoma. This project presents an enhanced deep learning approach utilizing a lightweight boundary-assisted UNet (LB-UNet) architecture, integrated with a ResNet-101 encoder and balanced data handling strategies. By addressing challenges such as severe class imbalance and limited computational resources, the proposed method achieves superior segmentation and classification performance. The project offers a robust, interpretable, and efficient framework for skin lesion analysis, paving the way for real-world clinical applications.

## Project Metadata
### Authors
- **Team:** Mohammed Nazmul Arefin
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** King Fahd University of Petroleum and Minerals (KFUPM)

### Project Documents
- **Presentation:** [Project Presentation](/https://kfupmedusa-my.sharepoint.com/:p:/g/personal/g202416760_kfupm_edu_sa/EVimCxdRs2JMk8McPLB0Ew8Bv-wmUDaj58u0EkdYLAM5ug?e=OGWPrt)
- **Report:** [Project Report](/https://kfupmedusa-my.sharepoint.com/:b:/g/personal/g202416760_kfupm_edu_sa/EcddL1_m1lBPrnKJ011rbagBloqdg83DRjNEfgUr_AfHFQ?e=WDODl1)

### Reference Paper
- [Skin Lesion Analysis Towards Melanoma Detection: A Challenge at ISIC 2020](https://challenge.isic-archive.com/data/)

### Reference Dataset
- [ISIC 2020 Challenge Dataset](https://challenge.isic-archive.com/data/)

## Project Technicalities

### Terminologies
- **Skin Lesion Segmentation:** The process of identifying and isolating regions of interest (lesions) from skin images.
- **LB-UNet:** A lightweight UNet architecture enhanced with boundary assistance for better lesion boundary detection.
- **ResNet-101:** A deep convolutional network with residual connections used as the encoder backbone.
- **Binary Cross-Entropy Loss:** A loss function suitable for binary classification problems.
- **Balanced Dataset:** A dataset with approximately equal numbers of benign and malignant samples to mitigate bias.
- **Early Stopping:** A technique to terminate training when performance ceases to improve, preventing overfitting.
- **ROC Curve:** A graphical plot that illustrates the diagnostic ability of a binary classifier system.
- **AUC (Area Under the Curve):** A performance metric summarizing the ROC curve.
- **Confusion Matrix:** A table used to evaluate the performance of a classification model.
- **Threshold Tuning:** Adjusting the decision threshold to optimize performance metrics.

### Problem Statements
- **Problem 1:** Severe class imbalance in skin lesion datasets leads to biased model predictions.
- **Problem 2:** Traditional models are resource-intensive and unsuitable for edge deployment.
- **Problem 3:** Existing architectures struggle to achieve robust boundary localization for lesions.

### Loopholes or Research Areas
- **Interpretability:** Lack of transparency in model decision-making.
- **Domain Shift:** Poor generalization to unseen data with different acquisition settings.
- **Fairness:** Risk of demographic bias across different skin tones and populations.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Balanced Sampling:** Select a subset of benign cases and combine them with all malignant samples.
2. **Lightweight Architecture:** Implement LB-UNet with a ResNet-101 encoder to enhance feature extraction while keeping the model efficient.
3. **Threshold Optimization:** Conduct precision-recall threshold tuning to balance sensitivity and specificity.

### Proposed Solution: Code-Based Implementation
The project provides a full deep learning pipeline in PyTorch:

- **LB-UNet Model:** A lightweight UNet modified with ResNet-101 backbone.
- **Balanced Dataset Loader:** Custom PyTorch class to sample and preprocess images.
- **Training Script:** Implements early stopping, logging, and model checkpointing.
- **Evaluation Script:** Computes Accuracy, AUC, Dice Score, IoU, and confusion matrices.

### Key Components
- **`dataset.py`**: Loads and preprocesses the balanced dataset.
- **`model.py`**: Defines the LB-UNet architecture with ResNet-101 encoder.
- **`train.py`**: Handles the training loop, logging, and early stopping.
- **`evaluate.py`**: Evaluates the model and generates plots.

## Model Workflow

1. **Input:**
   - **Images:** Preprocessed dermoscopic images resized to 124x124 pixels.
   - **Masks:** Binary segmentation masks for lesions (1 for melanoma, 0 for benign).

2. **Dataset Balancing:**
   - Randomly select 500 benign images.
   - Combine with all malignant images.
   - Split into 70% training, 15% validation, and 15% testing sets.

3. **Training Phase:**
   - Train LB-UNet using Adam optimizer and BCEWithLogitsLoss.
   - Monitor validation loss with early stopping (patience = 3 epochs).

4. **Testing Phase:**
   - Evaluate model performance using Accuracy, AUC, Dice Score, IoU.
   - Generate confusion matrix and ROC curves.

5. **Threshold Tuning:**
   - Adjust classification thresholds to optimize precision, recall, and F1-score.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/skin-cancer-lbunet.git
    cd skin-cancer-lbunet
    ```

2. **Set Up the Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Prepare the Dataset:**
    - Download the ISIC 2020 dataset.
    - Organize images and metadata as required.

4. **Train the Model:**
    ```bash
    python train.py --config configs/train_config.yaml
    ```

5. **Evaluate the Model:**
    ```bash
    python evaluate.py --checkpoint path/to/best_model.pt
    ```

## Acknowledgments
- **Open-Source Libraries:** PyTorch, Albumentations, Scikit-learn, TensorBoard.
- **Dataset Providers:** ISIC Challenge for providing dermoscopic image data.
- **Supervision:** Special thanks to Dr. Muzammil Behzad for mentorship and technical guidance.
- **Resource Support:** Gratitude to KFUPM for computational resources and project support.

