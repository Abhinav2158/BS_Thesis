import os
import numpy as np
import torch
import tqdm
from PIL import Image
import data_loader
from torch.utils import data
import torch.nn.functional as F
from tools.tools import calculate_psnr, calculate_ssim, calculate_ae ,calculate_lpips
from tools.MS_SWD import MS_SWD
from models import CycleGanNIR_net
from torch.cuda.amp import autocast  

def create_feathering_mask(patch_size, overlap):
    if overlap <= 0:
        return torch.ones(patch_size, patch_size)
    ramp = torch.linspace(0, 1, steps=overlap)
    center_length = patch_size - 2 * overlap
    if center_length < 0:
        center_length = 0
    flat_center = torch.ones(center_length)
    mask_1d = torch.cat([ramp, flat_center, torch.flip(ramp, dims=[0])])
    mask_2d = torch.ger(mask_1d, mask_1d)
    return mask_2d

def split_tensor_into_patches(tensor, patch_size=256, overlap=0):
    C, H, W = tensor.shape
    pad_bottom = max(0, patch_size - H)
    pad_right = max(0, patch_size - W)
    if pad_bottom > 0 or pad_right > 0:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
        H, W = tensor.shape[1], tensor.shape[2]
    stride = patch_size - overlap
    y_positions = list(range(0, H - patch_size + 1, stride))
    if not y_positions or y_positions[-1] != H - patch_size:
        y_positions.append(H - patch_size)
    x_positions = list(range(0, W - patch_size + 1, stride))
    if not x_positions or x_positions[-1] != W - patch_size:
        x_positions.append(W - patch_size)
    patches = []
    positions = []
    for y in y_positions:
        for x in x_positions:
            patch = tensor[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    return patches, positions

def get_patches(tensor, patch_size=256, overlap=0):
    _, H, W = tensor.shape
    if H < patch_size or W < patch_size:
        pad_bottom = max(0, patch_size - H)
        pad_right = max(0, patch_size - W)
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
        return [tensor]
    else:
        patches, _ = split_tensor_into_patches(tensor, patch_size, overlap)
        return patches

def merge_patches_feathered(patches, positions, image_shape, patch_size=256, overlap=0, device='cuda'):
    C, H, W = image_shape
    output = torch.zeros((C, H, W), device=device)
    weight = torch.zeros((1, H, W), device=device)
    feather_2d = create_feathering_mask(patch_size, overlap).to(device)
    feather_mask = feather_2d.unsqueeze(0)
    for patch, (y, x) in zip(patches, positions):
        weighted_patch = patch * feather_mask
        output[:, y:y+patch_size, x:x+patch_size] += weighted_patch
        weight[:, y:y+patch_size, x:x+patch_size] += feather_mask
    output = output / weight.clamp(min=1e-8)
    return output

def sliding_window_inference_pair(model, image1, image2, patch_size=256, overlap=32, device='cuda'):
    image1 = image1.squeeze(0)
    image2 = image2.squeeze(0)
    _, H, W = image1.shape
    patches1, positions = split_tensor_into_patches(image1, patch_size, overlap)
    patches2, _ = split_tensor_into_patches(image2, patch_size, overlap)
    
    processed_patches = []
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            for p1, p2 in zip(patches1, patches2):
                p1 = p1.unsqueeze(0).to(device)
                p2 = p2.unsqueeze(0).to(device)
                output_patch = model(p1, p2)
                if isinstance(output_patch, tuple):
                    output_patch = output_patch[1]
                output_patch = output_patch.squeeze(0)
                processed_patches.append(output_patch)
    
    merged_output = merge_patches_feathered(
        processed_patches, positions, (3, H, W),
        patch_size, overlap, device
    )
    return merged_output.unsqueeze(0)

if __name__ == '__main__':
    checkpoint_path = './knowledge_distillation/RGB2NIR_weights/best.pth'
    results_dir = './Results_&_weights/results_try'
    os.makedirs(results_dir, exist_ok=True)
    
    device = 'cuda'
    patch_size = 256
    overlap = 64  
    
    
    model = CycleGanNIR_net.all_Generator(3, 3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
   
    test_files = data_loader.get_test_paths()
    test_dataset = data_loader.Dataset_test(test_files, target_shape=None)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    #ms_swd_model = MS_SWD(num_scale=5, num_proj=128).to(device)
    
    psnr_list, ssim_list, ae_list, ms_swd_list ,lpips_list = [] ,[], [], [], []
    
    for i, batch in enumerate(tqdm.tqdm(test_loader, desc="Testing")):
        nir_gray = batch['nir_gray'].to(device)
        nir_hsv = batch['nir_hsv'].to(device)
        real_rgb = batch['rgb_rgb'].to(device)
        
        fake_rgb = sliding_window_inference_pair(model, nir_gray, nir_hsv, patch_size, overlap, device)
        
        real_rgb_np = real_rgb.cpu().numpy()[0].transpose(1, 2, 0)
        fake_rgb_np = fake_rgb.cpu().numpy()[0].transpose(1, 2, 0)
        
        psnr_val = calculate_psnr(real_rgb_np, fake_rgb_np)
        ssim_val = calculate_ssim(real_rgb_np, fake_rgb_np)
        ae_val = calculate_ae(real_rgb_np, fake_rgb_np)
        #ms_swd_val = ms_swd_model(fake_rgb, real_rgb).item()
        # lpips_val = calculate_lpips(fake_rgb, real_rgb).item()  # LPIPS expects torch tensors in range [-1, 1]
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        ae_list.append(ae_val)
        # ms_swd_list.append(ms_swd_val)
        # lpips_list.append(lpips_val)
        
        out_img = np.clip(fake_rgb_np, 0.0, 1.0)  # Make sure values are within [0, 1]
        out_img = np.nan_to_num(out_img)         # Replace NaNs/Infs with 0s
        out_img = (out_img * 255).astype(np.uint8)

        image_filename = os.path.join(results_dir, f'Test_{i+1:06d}.png')
        try:
            Image.fromarray(out_img).save(image_filename)
        except Exception as e:
            print(f"Error saving image {image_filename}: {e}")

    print("Average PSNR:", np.mean(psnr_list))
    print("Average SSIM:", np.mean(ssim_list))
    print("Average AE:", np.mean(ae_list))
    # print("Average MS-SWD:", np.mean(ms_swd_list))
    # print("Average LPIPS:", np.mean(lpips_list))

    
    with open('best_test.txt', 'w') as f:
        f.write("Average PSNR: %f\n" % np.mean(psnr_list))
        f.write("Average SSIM: %f\n" % np.mean(ssim_list))
        f.write("Average AE: %f\n" % np.mean(ae_list))
        # f.write("Average MS-SWD: %f\n" % np.mean(ms_swd_list))
        # f.write("Average LPIPS: %f\n\n" % np.mean(lpips_list))
        f.write("Detailed Metrics per Image:\n")
        for i, (psnr_val, ssim_val, ae_val) in enumerate(zip(psnr_list, ssim_list, ae_list, lpips_list)):
            f.write("Image %d: PSNR = %f, SSIM = %f, AE = %f, LPIPS = %f\n" % 
                    (i+1, psnr_val, ssim_val, ae_val))
