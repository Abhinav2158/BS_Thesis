# Impact of Various Loss Functions on NIR Image Colorization

## Overview

This folder, `folder_2_impact_of_various_loss_functions`, contains the implementation, results, and analysis of experiments conducted to enhance the visual quality of near-infrared (NIR) image colorization using the ColorMamba framework. The focus of this experiment was to address a key limitation observed in the original ColorMamba implementation: blurriness in fine textures and edges, which can hinder interpretability in critical applications such as autonomous driving and surveillance. By exploring alternative loss functions, this experiment aimed to improve the realism, sharpness, and color accuracy of colorized NIR-to-RGB images, aligning them more closely with human visual perception and practical requirements.

The experiment is part of my Bachelor of Science thesis in Data Science and Engineering at IISER Bhopal, conducted under the supervision of Dr. Samiran Das. All code, datasets, and results are organized here to ensure reproducibility and clarity for researchers, students, or practitioners interested in NIR colorization.

## Background

The ColorMamba framework transforms grayscale NIR images into RGB outputs using a dual-network architecture: an RGB Reconstruction Network (G_A) for generating the final colorized image and an HSV Color Prediction Sub-network (G_B) for guiding accurate color mapping. The original implementation relied on a combination of loss functions—Mean Squared Error (MSE), Cosine Similarity, and Multi-Scale Structural Similarity (MS-SSIM)—to optimize pixel-level accuracy and structural fidelity. While effective for achieving high quantitative metrics like Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM), this approach often produced outputs with noticeable blurriness, particularly in regions with intricate details such as foliage, road markings, or facial features.

Loss functions act as the guiding principles for training deep learning models, defining how errors between predicted and ground truth images are measured and minimized. The choice of loss function directly impacts the model’s ability to prioritize certain aspects of image quality, such as numerical precision, texture sharpness, or perceptual realism. This experiment investigates whether alternative loss functions can mitigate blurriness, enhance texture clarity, and produce more visually coherent outputs without sacrificing quantitative performance.

## Objective

The primary objective was to improve the perceptual quality of ColorMamba’s NIR-to-RGB colorized images by experimenting with advanced loss functions. Specifically, the experiment sought to:
- Reduce blurriness in detailed regions to enhance interpretability for applications requiring precise visual cues.
- Achieve a balance between numerical accuracy (e.g., low pixel errors) and visual realism (e.g., natural textures and colors).
- Identify a loss function configuration that aligns closely with human perception, making outputs more suitable for real-world scenarios like autonomous navigation.

This aligns with Research Question 1 (RQ1) from the thesis: *Enhance Image Quality through Advanced Loss Functions*.

## Methodology

The experiment was conducted using the VCIP2020 dataset, which provides 372 paired NIR-RGB images for training and 37 for validation. The ColorMamba model was trained for 50 epochs with a batch size of 8, following the original training strategy, but with modifications to the loss function. Below, I detail the loss functions tested and the experimental setup.

### Loss Functions Explored

1. **Original Loss (Baseline)**:
   - **Components**: Mean Squared Error (MSE), Cosine Similarity, and Multi-Scale SSIM (MS-SSIM).
   - **Purpose**: MSE ensures pixel-wise accuracy, Cosine Similarity aligns feature directions, and MS-SSIM captures structural similarity across scales.
   - **Formulation**:
     \[
     \mathcal{L}_{\text{original}} = \lambda_{\text{mse}} \mathcal{L}_{\text{mse}} + \lambda_{\text{cosine}} \mathcal{L}_{\text{cosine}} + \lambda_{\text{msssim}} \mathcal{L}_{\text{msssim}}
     \]
     where \(\mathcal{L}_{\text{mse}} = \frac{1}{n} \sum_{i=1}^n (x_i - y_i)^2\), and weights \(\lambda\) balance contributions.
   - **Rationale**: This combination was designed for numerical precision but often led to over-smoothed outputs, motivating the exploration of alternatives.

2. **VGG Perceptual Loss**:
   - **Components**: Feature-based loss using a pre-trained VGG-19 network.
   - **Purpose**: Captures high-level content and texture information by comparing feature maps from intermediate VGG layers, prioritizing perceptual similarity over pixel-wise differences.
   - **Formulation**:
     \[
     \mathcal{L}_{\text{vgg}} = \sum_{l} \frac{1}{N_l} \|\phi_l(\hat{y}) - \phi_l(y)\|_2^2
     \]
     where \(\phi_l\) denotes features from layer \(l\), and \(\hat{y}\), \(y\) are predicted and ground truth images.
   - **Rationale**: Encourages outputs that align with human perception, emphasizing textures and structures critical for realistic rendering.

3. **Histogram Loss**:
   - **Components**: Loss based on matching the color histogram distributions of predicted and ground truth images.
   - **Purpose**: Ensures global color consistency (e.g., brightness and tone distribution), preventing drastic color shifts across the image.
   - **Formulation**: Computes the difference between histograms of RGB channels, typically using a divergence metric like Kullback-Leibler or Chi-squared.
   - **Rationale**: Complements VGG loss by regularizing overall color balance, which is essential for natural-looking images.

4. **Combined Loss**:
   - **Components**: Integrates VGG Perceptual Loss, Histogram Loss, and Cosine Similarity Loss.
   - **Purpose**: Balances local detail preservation (VGG), global color consistency (Histogram), and feature alignment (Cosine).
   - **Formulation**:
     \[
     \mathcal{L}_{\text{combined}} = \lambda_{\text{vgg}} \mathcal{L}_{\text{vgg}} + \lambda_{\text{hist}} \mathcal{L}_{\text{hist}} + \lambda_{\text{cosine}} \mathcal{L}_{\text{cosine}}
     \]
     with tuned weights \(\lambda\) to optimize trade-offs.
   - **Rationale**: Aims to leverage the strengths of each loss to produce images that are both numerically accurate and visually appealing.

### Experimental Setup

- **Dataset**: VCIP2020 (256x256 resolution, 372 training pairs, 37 validation pairs).
- **Preprocessing**: Images were resized to 256x256 using bicubic interpolation, randomly cropped to 200x200 during training, and normalized to [0,1].
- **Training**: The ColorMamba model was trained for 50 epochs using the Adam optimizer, with a learning rate adjusted every 25 epochs. Each loss function configuration was tested independently.
- **Evaluation Metrics**:
  - **Peak Signal-to-Noise Ratio (PSNR)**: Measures image clarity (higher is better).
  - **Structural Similarity Index (SSIM)**: Assesses structural fidelity (higher is better).
  - **Absolute Error (AE)**: Quantifies pixel-wise differences (lower is better).
  - Qualitative visual inspections were also conducted to assess texture sharpness and color realism.
- **Implementation**: Code was written in PyTorch, leveraging pre-trained VGG-19 models from torchvision for perceptual loss and custom histogram loss implementations.

