# Model Improvements - DCN-V2 with Transformer and MaskNet Baseline

## Overview

In this work, we propose an improved version of the DCN-V2 model by replacing the CrossNet module with a Transformer. The Transformer helps learn more complex relationships between features by leveraging self-attention mechanisms, which improves the model's ability to capture both **explicit** and **implicit** feature interactions. This is particularly beneficial for sequential data or data with complex inter-feature dependencies.

Additionally, we evaluated the performance of **MaskNet** as a baseline for comparison, providing an extensive evaluation of various models with different architectures.

## Model Architecture

### DCN-V2 with Transformer
- **CrossNet Replacement**: We replaced the CrossNet module in the original DCN-V2 model with a Transformer to improve the ability to capture higher-order feature interactions. The Transformer utilizes self-attention to learn complex dependencies between features.
- **Feature Interaction Learning**: By leveraging explicit interaction (via CrossNet) and implicit interaction (via Transformer), we aim to improve the model's overall performance.
- **Flexible Model Architecture**: The model supports various architectures, including "crossnet only", "stacked", "parallel", and "stacked parallel", enabling more flexible and efficient optimization.

### MaskNet Baseline
- We also evaluate the performance of **MaskNet** with different baseline models, which serves as a reference for the improvements made.

## Results

Below is a comparison of various models and their respective **AUC** scores on the Criteo dataset.

| **Model**       | **AUC**  |
|-----------------|----------|
| **Criteo**      | -        |
| **PNN**         | 0.8099   |
| **DeepFM**      | 0.8099   |
| **DLRM**        | 0.8092   |
| **xDeepFM**     | 0.8099   |
| **AutoInt+**    | 0.8101   |
| **DCN**         | 0.8099   |
| **DNN**         | 0.8098   |
| **SerMaskNet**  | 0.8119   |
| **DCN-V2**      | 0.8115   |
| **Ours**        | **0.8135** |

The improved model (**Ours**) demonstrates the best AUC score of **0.8135**, outperforming all other baseline models, including DCN-V2.

## References

- **Google Drive**: [Link to Google Drive](<Insert your Google Drive link here>)
- **GitHub Repository**: [Link to GitHub Repository](https://github.com/reczoo/FuxiCTR)
