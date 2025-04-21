import os
import time
import numpy as np
import torch
import tqdm
from PIL import Image
import data_loader
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
from tools.tools import calculate_psnr, calculate_ssim, calculate_ae
from CycleGanNIR_net_kd import all_Generator, all_Generator_Small
from torch.amp import autocast
import matplotlib.pyplot as plt

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def sliding_window_inference_pair(model, image1, image2, patch_size=256, overlap=128, device='cuda', batch_size=2):
    image1 = image1.squeeze(0)
    image2 = image2.squeeze(0)
    _, orig_H, orig_W = image1.shape
    
    # Pad to nearest multiple of patch_size
    target_H = ((orig_H + patch_size - 1) // patch_size) * patch_size
    target_W = ((orig_W + patch_size - 1) // patch_size) * patch_size
    pad_bottom = target_H - orig_H
    pad_right = target_W - orig_W
    if pad_bottom > 0 or pad_right > 0:
        image1 = F.pad(image1, (0, pad_right, 0, pad_bottom), mode='reflect')
        image2 = F.pad(image2, (0, pad_right, 0, pad_bottom), mode='reflect')
    H, W = image1.shape[1], image2.shape[2]
    
    stride = patch_size - overlap
    patches1 = image1.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches1 = patches1.contiguous().view(1, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    patches2 = image2.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches2 = patches2.contiguous().view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    
    positions = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            positions.append((y, x))
    
    processed_patches = []
    model.eval()
    with torch.no_grad():
        with autocast(device_type='cuda', enabled=True):
            for i in range(0, len(patches1), batch_size):
                batch_patches1 = patches1[i:i+batch_size]  # This is already a list of tensors
                batch_patches2 = patches2[i:i+batch_size]
                if len(batch_patches1) == 0:  # Skip empty batches
                    continue
                batch_p1 = torch.stack(list(batch_patches1)).to(device)  # Ensure it's a list
                batch_p2 = torch.stack(list(batch_patches2)).to(device)
                output_batch = model(batch_p1, batch_p2)
                if isinstance(output_batch, tuple):
                    output_batch = output_batch[1]
                processed_patches.extend([p for p in output_batch])
                # Clear GPU memory after each patch batch
                torch.cuda.empty_cache()
    
    # Merge patches
    C, H_out, W_out = 3, H, W
    output = torch.zeros((C, H_out, W_out), device=device)
    weight = torch.zeros((1, H_out, W_out), device=device)
    feather_mask = create_feathering_mask(patch_size, overlap, device)
    
    for patch, (y, x) in zip(processed_patches, positions):
        y_end = min(y + patch_size, H_out)
        x_end = min(x + patch_size, W_out)
        patch_h, patch_w = y_end - y, x_end - x
        weighted_patch = patch[:, :patch_h, :patch_w] * feather_mask[:, :patch_h, :patch_w]
        output[:, y:y_end, x:x_end] += weighted_patch
        weight[:, y:y_end, x:x_end] += feather_mask[:, :patch_h, :patch_w]
    
    output = output / weight.clamp(min=1e-8)
    output = torch.clamp(output, 0, 1)
    
    # Crop to original size
    output = output[:, :orig_H, :orig_W]
    return output.unsqueeze(0)

def create_feathering_mask(patch_size, overlap, device='cuda'):
    if overlap <= 0:
        return torch.ones(1, patch_size, patch_size, device=device)
    
    # Create a 1D feathering ramp with a smoother transition
    ramp = torch.linspace(0, 1, steps=overlap, device=device) ** 2  # Quadratic ramp for smoother blending
    center_length = patch_size - 2 * overlap
    if center_length < 0:
        center_length = 0
        ramp = torch.linspace(0, 1, steps=patch_size // 2, device=device) ** 2
    flat_center = torch.ones(center_length, device=device)
    mask_1d = torch.cat([ramp, flat_center, torch.flip(ramp, dims=[0])])
    
    # Ensure the mask_1d has the correct length
    mask_1d = mask_1d[:patch_size]
    if len(mask_1d) < patch_size:
        mask_1d = F.pad(mask_1d, (0, patch_size - len(mask_1d)), value=1.0)
    
    # Create a 2D mask
    mask_2d = torch.outer(mask_1d, mask_1d)
    return mask_2d.unsqueeze(0)

if __name__ == '__main__':
    teacher_checkpoint = './Results_&_weights/weights_exp5_1.5hist_OMSIV/best.pth'
    student_checkpoint = './knowledge_distillation/weights_kd_OMSIV/student_best.pth'
    results_dir = './knowledge_distillation/results_kd_OMSIV_new'
    os.makedirs(results_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not os.path.exists(student_checkpoint):
        raise FileNotFoundError(f"Student checkpoint not found at {student_checkpoint}. Please run training first.")
    
    patch_size = 256
    overlap = 128  # Increased for better blending, aligned with train_kd.py
    patch_batch_size = 2  # Reduced for memory management
    
    teacher_model = all_Generator(3, 3).to(device)
    teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
    teacher_model.eval()
    
    student_model = all_Generator_Small(3, 3).to(device)
    student_model.load_state_dict(torch.load(student_checkpoint, map_location=device))
    student_model.eval()
    
    test_files = data_loader.get_test_paths()
    test_dataset = data_loader.Dataset_test(test_files)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    teacher_psnr = []
    teacher_ssim = []
    teacher_ae = []
    teacher_times = []
    student_psnr = []
    student_ssim = []
    student_ae = []
    student_times = []
    
    for i, batch in enumerate(tqdm.tqdm(test_loader, desc="Testing")):
        nir_gray = batch['nir_gray'].to(device, non_blocking=True)
        nir_hsv = batch['nir_hsv'].to(device, non_blocking=True)
        real_rgb = batch['rgb_rgb'].to(device, non_blocking=True)
        original_size = batch['original_size']  # (width, height)
        
        start_time = time.time()
        teacher_fake_rgb = sliding_window_inference_pair(teacher_model, nir_gray, nir_hsv, patch_size, overlap, device, batch_size=patch_batch_size)
        teacher_time = (time.time() - start_time) * 1000
        teacher_times.append(teacher_time)
        
        start_time = time.time()
        student_fake_rgb = sliding_window_inference_pair(student_model, nir_gray, nir_hsv, patch_size, overlap, device, batch_size=patch_batch_size)
        student_time = (time.time() - start_time) * 1000
        student_times.append(student_time)
        
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Clear memory after processing each image
        
        real_rgb_np = real_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)
        teacher_fake_rgb_np = teacher_fake_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)
        student_fake_rgb_np = student_fake_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)
        
        teacher_psnr.append(calculate_psnr(real_rgb_np, teacher_fake_rgb_np))
        teacher_ssim.append(calculate_ssim(real_rgb_np, teacher_fake_rgb_np))
        teacher_ae.append(calculate_ae(real_rgb_np, teacher_fake_rgb_np))
        
        student_psnr.append(calculate_psnr(real_rgb_np, student_fake_rgb_np))
        student_ssim.append(calculate_ssim(real_rgb_np, student_fake_rgb_np))
        student_ae.append(calculate_ae(real_rgb_np, student_fake_rgb_np))
        
        out_img = (teacher_fake_rgb_np * 255).astype(np.uint8)
        image_filename = os.path.join(results_dir, f'teacher_result_{i+1}.png')
        Image.fromarray(out_img).save(image_filename)

        out_img = (student_fake_rgb_np * 255).astype(np.uint8)
        image_filename = os.path.join(results_dir, f'result_{i+1}.png')
        Image.fromarray(out_img).save(image_filename)
    
    print("Teacher - Average PSNR:", np.mean(teacher_psnr))
    print("Teacher - Average SSIM:", np.mean(teacher_ssim))
    print("Teacher - Average AE:", np.mean(teacher_ae))
    print("Teacher - Average Inference Time:", np.mean(teacher_times), "ms")
    
    print("Student - Average PSNR:", np.mean(student_psnr))
    print("Student - Average SSIM:", np.mean(student_ssim))
    print("Student - Average AE:", np.mean(student_ae))
    print("Student - Average Inference Time:", np.mean(student_times), "ms")
    
    with open('best_test_kd.txt', 'w') as f:
        f.write(f"Teacher - Average PSNR: {np.mean(teacher_psnr)}\n")
        f.write(f"Teacher - Average SSIM: {np.mean(teacher_ssim)}\n")
        f.write(f"Teacher - Average AE: {np.mean(teacher_ae)}\n")
        f.write(f"Teacher - Average Inference Time: {np.mean(teacher_times)} ms\n")
        f.write(f"Student - Average PSNR: {np.mean(student_psnr)}\n")
        f.write(f"Student - Average SSIM: {np.mean(student_ssim)}\n")
        f.write(f"Student - Average AE: {np.mean(student_ae)}\n")
        f.write(f"Student - Average Inference Time: {np.mean(student_times)} ms\n")
    
    quality_metrics = ['PSNR', 'SSIM', 'AE']
    teacher_quality_values = [np.mean(teacher_psnr), np.mean(teacher_ssim), np.mean(teacher_ae)]
    student_quality_values = [np.mean(student_psnr), np.mean(student_ssim), np.mean(student_ae)]

    x = np.arange(len(quality_metrics))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(x - width/2, teacher_quality_values, width, label='Teacher', color='#1f77b4')
    ax1.bar(x + width/2, student_quality_values, width, label='Student', color='#17becf')
    ax1.set_ylabel('Values')
    ax1.set_title('Quality Metrics: PSNR, SSIM, AE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(quality_metrics)
    ax1.legend()

    ax2.bar(['Teacher', 'Student'], [np.mean(teacher_times), np.mean(student_times)], 
            color=['#9467bd', '#ff7f0e'], width=0.3)
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Inference Time')

    plt.tight_layout()
    plt.savefig('./knowledge_distillation/performance_comparison_two_plots_RGB2NIR.png')
    plt.show()