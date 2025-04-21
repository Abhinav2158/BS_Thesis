# NIR Image Colorization with ColorMamba

Welcome to my Bachelor of Science thesis project repository, where I explore near-infrared (NIR) image colorization using the ColorMamba framework. This README provides a beginner-friendly overview of the project, explaining what I aimed to achieve and how I approached it, without diving into results (those are detailed in the respective experiment folders). The goal is to help anyone, even reading for the first time, understand the purpose and structure of my work.

## Project Overview

Near-infrared (NIR) imaging captures light (700-1400 nm) beyond what our eyes can see, producing grayscale images that excel in tough conditions like low light, fog, or rain. These images are incredibly useful for applications such as autonomous driving (e.g., spotting obstacles at night), surveillance, and medical imaging. However, their lack of color makes them hard for humans to interpret and less intuitive for computer vision systems compared to vibrant RGB images we’re used to.

My project focuses on transforming these grayscale NIR images into realistic, colorful RGB images using a deep learning framework called ColorMamba. Think of ColorMamba as a smart artist that learns to add colors—like green for grass or red for a car—while keeping every detail sharp, from the big picture (like a road) to tiny textures (like pebbles). The challenge is to make these images look natural, work across different scenes (e.g., forests, cities, or faces), and do it efficiently enough for real-time use, like in a self-driving car that needs to react instantly.

### Why This Matters

Colorizing NIR images bridges the gap between their technical strengths and practical usability. For example:
- In autonomous driving, colored images help identify road signs or pedestrians clearly, even in darkness.
- In surveillance, they make it easier to spot details in foggy conditions.
- In medical imaging, they could enhance visualization of tissues captured in NIR.

My work aims to improve ColorMamba to produce high-quality, reliable colorized images that can be used in these real-world scenarios, addressing challenges like limited training data, computational demands, and varying environments.

### How ColorMamba Works

ColorMamba is a deep learning model with two key parts working together:
1. **RGB Reconstruction Network**: This is the main component, taking a grayscale NIR image and generating a full-color RGB version. It uses a U-Net-like structure, which first shrinks the image to understand the overall scene (e.g., a park layout) and then expands it to add details (e.g., leaves on trees). A special feature called the Visual State Space Block (VSSB) helps it capture both the big picture and fine details.
2. **HSV Color Prediction Sub-network**: This acts as a color guide, predicting a color map in HSV (Hue, Saturation, Value) format to help the main network choose accurate colors. For instance, it ensures skies are blue and not purple.

These components are trained on paired NIR-RGB images (like those in the VCIP2020 dataset), learning to map grayscale inputs to colorful outputs. The model is guided by “loss functions” (scorecards) to minimize errors and improve realism, and its performance is evaluated using metrics like PSNR (clarity), SSIM (structural similarity), and AE (pixel errors).

## Experiments: What Did I Explore?

To make ColorMamba better, I conducted experiments targeting specific challenges in NIR colorization. These are organized into three main parts, each with its own folder containing detailed results and code. Below, I describe what each experiment aimed to achieve, keeping it clear and focused on the approach rather than outcomes.

### Part 1: Reproducibility of Results

**Goal**: Verify that ColorMamba performs as expected before improving it, ensuring a solid foundation.
**Approach**: I implemented the original ColorMamba framework using the VCIP2020 dataset, which provides paired NIR-RGB images. I trained the model with its default setup, including its mix of loss functions (Mean Squared Error, Cosine Similarity, and Multi-Scale SSIM), to see how well it colorizes images out of the box. This step was like test-driving a car to confirm it runs smoothly before adding upgrades.
**Where to Find Details**: Check `folder_1_reproducibility_of_results` for code and findings.

### Part 2: Impact of Various Loss Functions

**Goal**: Improve the visual quality of ColorMamba’s images, especially to reduce blurriness in detailed areas like textures or edges, which is critical for applications like autonomous driving.
**Approach**: I experimented with new loss functions to guide the model better:
- **VGG Perceptual Loss**: Compares high-level features (like shapes and textures) using a pre-trained VGG network, aiming for more natural-looking images.
- **Histogram Loss**: Ensures the overall color distribution (e.g., brightness and tones) matches the real image.
- **Combined Loss**: Blends VGG, histogram, and cosine similarity losses to balance detail clarity and color accuracy.
I tested these on the VCIP2020 dataset, comparing them to the original loss setup to see which produces sharper, more realistic images.
**Where to Find Details**: See `folder_2_impact_of_various_loss_functions` for scripts and observations.

### Part 3: Robustness Across Multiple Datasets

**Goal**: Make ColorMamba versatile enough to colorize images from different scenarios, not just the VCIP2020 dataset, so it works in varied real-world settings.
**Approach**: I tested the model on four datasets with different characteristics:
- **FANVID**: High-resolution facial images.
- **OMSIV**: Outdoor scenes like roads and fields.
- **RGB2NIR Scene**: Varied environments (forests, cities, etc.).
- **VCIP2020**: Indoor and outdoor settings.
Since these datasets have larger image sizes than ColorMamba’s default 256x256, resizing caused blurriness. Instead, I developed a **patch-based method**, splitting images into 256x256 patches, processing each one, and stitching them back together smoothly. I also added a preprocessing step for very large images to avoid repetitive patches (like plain sky), ensuring the model learns diverse features.
**Where to Find Details**: Explore `folder_3_robustness_of_multiple_datasets` for code and comparisons.

## Repository Structure

The project is organized into folders, each corresponding to an experiment, plus additional folders for future work and conclusions:

- **folder_1_reproducibility_of_results**: Code and documentation for verifying ColorMamba’s baseline performance.
- **folder_2_impact_of_various_loss_functions**: Scripts and notes on testing new loss functions to enhance image quality.
- **folder_3_robustness_of_multiple_datasets**: Implementation and analysis of patch-based processing across multiple datasets.
- **folder_4_faster_inference**: (Planned) Will contain optimizations to speed up ColorMamba for real-time use (e.g., mixed precision, quantization).
- **folder_5_conclusion**: Includes a README summarizing the project’s key takeaways and future plans.

## Installation

To try the code yourself:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nir-colorization.git
