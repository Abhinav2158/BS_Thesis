import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import tqdm
from PIL import Image
import data_loader
from torch.utils import data
from tools.tools import calculate_psnr
from CycleGanNIR_net_kd import all_Generator, all_Generator_Small, NLayerDiscriminator
from torch.amp import autocast, GradScaler
from torchvision.models import vgg16

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def custom_collate_fn(batch):
    return batch

def split_tensor_into_patches(tensor, patch_size=256, overlap=0):
    C, orig_H, orig_W = tensor.shape
    
    # Pad to nearest multiple of patch_size
    target_H = ((orig_H + patch_size - 1) // patch_size) * patch_size
    target_W = ((orig_W + patch_size - 1) // patch_size) * patch_size
    pad_bottom = target_H - orig_H
    pad_right = target_W - orig_W
    if pad_bottom > 0 or pad_right > 0:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
    
    H, W = tensor.shape[1], tensor.shape[2]
    stride = patch_size - overlap
    patches = tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.contiguous().view(C, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    
    positions = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            positions.append((y, x))
    
    return list(patches), positions, (orig_H, orig_W)

def get_patches(tensor, patch_size=256, overlap=0):
    _, H, W = tensor.shape
    if H < patch_size or W < patch_size:
        pad_bottom = max(0, patch_size - H)
        pad_right = max(0, patch_size - W)
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
        return [tensor]
    else:
        patches, _, _ = split_tensor_into_patches(tensor, patch_size, overlap)
        return patches

def merge_patches_feathered(patches, positions, orig_size, patch_size=256, overlap=0, device='cuda'):
    C, orig_H, orig_W = 3, orig_size[0], orig_size[1]
    output = torch.zeros((C, orig_H, orig_W), device=device)
    weight = torch.zeros((1, orig_H, orig_W), device=device)
    feather_mask = create_feathering_mask(patch_size, overlap, device)
    
    for patch, (y, x) in zip(patches, positions):
        y_end = min(y + patch_size, orig_H)
        x_end = min(x + patch_size, orig_W)
        patch_h, patch_w = y_end - y, x_end - x
        
        # Ensure the patch and mask are correctly sized
        weighted_patch = patch[:, :patch_h, :patch_w] * feather_mask[:, :patch_h, :patch_w]
        output[:, y:y_end, x:x_end] += weighted_patch
        weight[:, y:y_end, x:x_end] += feather_mask[:, :patch_h, :patch_w]
    
    # Normalize the output, ensuring no division by zero
    weight = weight.clamp(min=1e-8)
    output = output / weight
    output = torch.clamp(output, 0, 1)
    
    return output

def sliding_window_inference_pair(model, image1, image2, patch_size=256, overlap=128, device='cuda', batch_size=8):
    image1 = image1.squeeze(0)
    image2 = image2.squeeze(0)
    
    # Split both images into patches
    patches1, positions, orig_size = split_tensor_into_patches(image1, patch_size, overlap)
    patches2, _, _ = split_tensor_into_patches(image2, patch_size, overlap)
    
    if len(patches1) != len(patches2):
        raise ValueError("Mismatch in number of patches between image1 and image2")
    
    processed_patches = []
    model.eval()
    with torch.no_grad():
        with autocast(device_type='cuda', enabled=True):
            for i in range(0, len(patches1), batch_size):
                batch_p1 = torch.stack(patches1[i:i+batch_size]).to(device)
                batch_p2 = torch.stack(patches2[i:i+batch_size]).to(device)
                output_batch = model(batch_p1, batch_p2)
                if isinstance(output_batch, tuple):
                    output_batch = output_batch[1]
                processed_patches.extend([p for p in output_batch])
    
    # Merge the processed patches
    merged_output = merge_patches_feathered(processed_patches, positions, orig_size, patch_size, overlap, device)
    return merged_output.unsqueeze(0)

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
    checkpoint_dir = './knowledge_distillation/weights_kd_RGB2NIR'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    gpu_ids = ['cuda:0']
    teacher_model = all_Generator(3, 3).to(gpu_ids[0])
    teacher_checkpoint = './RGB2NIR_weights/best.pth'
    teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=gpu_ids[0]))
    teacher_model.eval()
    
    student_model = all_Generator_Small(3, 3).to(gpu_ids[0])
    student_model.train()
    
    discriminator = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(gpu_ids[0])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Define projection layers outside the loop
    proj_layer_coder3 = nn.Conv2d(128, 256, 1).to(gpu_ids[0])  # Match coder_3
    proj_layer_coder5 = nn.Conv2d(128, 512, 3, stride=4, padding=1).to(gpu_ids[0])  # Match coder_5 spatial dimensions
    
    val_results_dir = './knowledge_distillation/validation_results_kd_RGB2NIR'
    os.makedirs(val_results_dir, exist_ok=True)
    
    train_files, val_files = data_loader.get_data_paths()
    train_dataset = data_loader.Dataset(train_files, target_shape=None)
    val_dataset = data_loader.Dataset(val_files, target_shape=None, return_name=True)
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=4,  # Reduced to manage memory
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    n_epochs = 50
    patch_size = 256
    overlap = 128  # Increased for better blending
    schedule = 5
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=schedule, gamma=0.5)
    criterion_distill = nn.MSELoss()
    criterion_supervised = nn.MSELoss()
    criterion_gan = nn.BCEWithLogitsLoss()  # Align with NLayerDiscriminator
    vgg = vgg16(weights='IMAGENET1K_V1').features.to(gpu_ids[0]).eval()
    scaler = GradScaler('cuda')
    
    best_psnr = -1.0
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_distill_loss = 0.0
        total_feature_loss = 0.0
        total_supervised_loss = 0.0
        total_percep_loss = 0.0
        total_gan_loss = 0.0
        total_patch_count = 0
        alpha = 0.9 if epoch < 30 else 0.5
        
        dt_size = len(train_loader.dataset)
        pbar = tqdm.tqdm(total=dt_size, desc=f'Epoch {epoch+1}/{n_epochs}', miniters=1)
        
        for batch in train_loader:
            for sample in batch:
                patches_nir_gray = get_patches(sample['nir_gray'], patch_size, overlap)
                patches_nir_hsv = get_patches(sample['nir_hsv'], patch_size, overlap)
                patches_rgb_rgb = get_patches(sample['rgb_rgb'], patch_size, overlap)
                
                num_patches = len(patches_nir_gray)
                patch_batch_size = 2  # Process 2 patches at a time to manage memory
                for idx in range(0, num_patches, patch_batch_size):
                    batch_patches_nir_gray = torch.stack(patches_nir_gray[idx:min(idx+patch_batch_size, num_patches)]).to(gpu_ids[0], non_blocking=True)
                    batch_patches_nir_hsv = torch.stack(patches_nir_hsv[idx:min(idx+patch_batch_size, num_patches)]).to(gpu_ids[0], non_blocking=True)
                    batch_patches_rgb_rgb = torch.stack(patches_rgb_rgb[idx:min(idx+patch_batch_size, num_patches)]).to(gpu_ids[0], non_blocking=True)
                    
                    with torch.no_grad():
                        with autocast(device_type='cuda', enabled=True):
                            _, teacher_output = teacher_model(batch_patches_nir_gray, batch_patches_nir_hsv)
                            teacher_netG_A = teacher_model.netG_A
                            x1_t = teacher_netG_A.coder_1(batch_patches_nir_gray)
                            x2_t = teacher_netG_A.coder_2(x1_t)
                            x3_t = teacher_netG_A.coder_3(x2_t)
                            x4_t = teacher_netG_A.coder_4(x3_t)
                            x5_t = teacher_netG_A.coder_5(x4_t)
                            teacher_features = [x3_t, x5_t]
                    
                    with autocast(device_type='cuda', enabled=True):
                        _, student_output = student_model(batch_patches_nir_gray, batch_patches_nir_hsv)
                        student_netG_A = student_model.netG_A
                        x1_s = student_netG_A.coder_1(batch_patches_nir_gray)
                        x2_s = student_netG_A.coder_2(x1_s)
                        x3_s = student_netG_A.coder_3(x2_s)
                        student_features = [proj_layer_coder3(x3_s), proj_layer_coder5(x3_s)]
                        
                        for sf, tf in zip(student_features, teacher_features):
                            if sf.shape != tf.shape:
                                print(f"Feature shape mismatch: student {sf.shape}, teacher {tf.shape}")
                                raise ValueError("Feature shapes must match for MSE loss")
                        
                        distill_loss = criterion_distill(student_output, teacher_output)
                        supervised_loss = criterion_supervised(student_output, batch_patches_rgb_rgb)
                        percep_loss = F.mse_loss(vgg(student_output)[:4], vgg(batch_patches_rgb_rgb)[:4])
                        feature_loss = sum(F.mse_loss(sf, tf) for sf, tf in zip(student_features, teacher_features))
                        gan_loss = criterion_gan(discriminator(student_output), torch.ones_like(discriminator(student_output)))
                        loss = alpha * (distill_loss + 0.1 * feature_loss) + (1 - alpha) * supervised_loss + 0.1 * percep_loss + 0.05 * gan_loss
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    with autocast(device_type='cuda', enabled=True):
                        real_pred = discriminator(batch_patches_rgb_rgb)
                        fake_pred = discriminator(student_output.detach())
                        d_loss = criterion_gan(real_pred, torch.ones_like(real_pred)) + criterion_gan(fake_pred, torch.zeros_like(fake_pred))
                    optimizer_d.zero_grad()
                    scaler.scale(d_loss).backward()
                    scaler.step(optimizer_d)
                    scaler.update()
                    
                    batch_size = batch_patches_nir_gray.size(0)
                    total_loss += loss.item() * batch_size
                    total_distill_loss += distill_loss.item() * batch_size
                    total_feature_loss += feature_loss.item() * batch_size
                    total_supervised_loss += supervised_loss.item() * batch_size
                    total_percep_loss += percep_loss.item() * batch_size
                    total_gan_loss += gan_loss.item() * batch_size
                    total_patch_count += batch_size
                    
                    # Clear GPU memory after each patch batch
                    torch.cuda.empty_cache()
            pbar.update(len(batch))
        pbar.close()
        
        scheduler.step()
        
        avg_loss = total_loss / total_patch_count
        avg_distill_loss = total_distill_loss / total_patch_count
        avg_feature_loss = total_feature_loss / total_patch_count
        avg_supervised_loss = total_supervised_loss / total_patch_count
        avg_percep_loss = total_percep_loss / total_patch_count
        avg_gan_loss = total_gan_loss / total_patch_count
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}, Distill Loss = {avg_distill_loss:.6f}, "
              f"Feature Loss = {avg_feature_loss:.6f}, Supervised Loss = {avg_supervised_loss:.6f}, "
              f"Perceptual Loss = {avg_percep_loss:.6f}, GAN Loss = {avg_gan_loss:.6f}")
        
        val_psnr_list = []
        teacher_psnr_list = []
        student_model.eval()
        with torch.no_grad():
            val_pbar = tqdm.tqdm(val_loader, desc=f'Validation Epoch {epoch+1}', leave=False)
            for batch in val_pbar:
                sample = batch[0]
                nir_gray = sample['nir_gray'].unsqueeze(0).to(gpu_ids[0], non_blocking=True)
                nir_hsv = sample['nir_hsv'].unsqueeze(0).to(gpu_ids[0], non_blocking=True)
                real_rgb = sample['rgb_rgb'].unsqueeze(0).to(gpu_ids[0], non_blocking=True)
                
                fake_rgb = sliding_window_inference_pair(student_model, nir_gray, nir_hsv, patch_size, overlap, gpu_ids[0])
                teacher_rgb = sliding_window_inference_pair(teacher_model, nir_gray, nir_hsv, patch_size, overlap, gpu_ids[0])
                
                real_rgb_np = real_rgb.cpu().numpy()[0].transpose(1, 2, 0)
                fake_rgb_np = fake_rgb.cpu().numpy()[0].transpose(1, 2, 0)
                teacher_rgb_np = teacher_rgb.cpu().numpy()[0].transpose(1, 2, 0)
                
                psnr_val = calculate_psnr(real_rgb_np, fake_rgb_np)
                teacher_psnr = calculate_psnr(real_rgb_np, teacher_rgb_np)
                val_psnr_list.append(psnr_val)
                teacher_psnr_list.append(teacher_psnr)
                
                fake_rgb_img = (fake_rgb_np * 255).astype(np.uint8)
                teacher_rgb_img = (teacher_rgb_np * 255).astype(np.uint8)
                base_name = os.path.splitext(os.path.basename(sample['rgb_path']))[0]
                image_filename = os.path.join(val_results_dir, f'epoch_{epoch+1}/generated_{base_name}.png')
                teacher_filename = os.path.join(val_results_dir, f'epoch_{epoch+1}/teacher_{base_name}.png')
                os.makedirs(os.path.dirname(image_filename), exist_ok=True)
                Image.fromarray(fake_rgb_img).save(image_filename)
                Image.fromarray(teacher_rgb_img).save(teacher_filename)
            
            val_pbar.close()
            avg_val_psnr = np.mean(val_psnr_list)
            avg_teacher_psnr = np.mean(teacher_psnr_list)
            print(f"Epoch {epoch+1}: Avg Validation PSNR = {avg_val_psnr:.4f}, Avg Teacher PSNR = {avg_teacher_psnr:.4f}")
        
        student_model.train()
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(student_model.state_dict(), os.path.join(checkpoint_dir, 'student_best.pth'))
        
        if epoch % 10 == 0:
            torch.save(student_model.state_dict(), os.path.join(checkpoint_dir, f'student_weights_{epoch}.pth'))