import cv2
from torch import nn
import numpy as np
import functools
import torch
from models.trans import CrissCrossAttention
from models.commom import *
import torch.nn.functional as F
from models.gen_net import UNetGenerator
from models.mambair_arch import MambaIR

class dsuGenerator_hsv(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(dsuGenerator_hsv, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.coder_1 = nn.Sequential(nn.Conv2d(3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias))
        self.coder_2 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2))
        self.coder_3 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 4))
        self.coder_4 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.coder_5 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.coder_6 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.coder_7 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.innermost_8 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_7 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_6 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_5 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_4 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 4))
        self.decoder_3 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2))
        self.decoder_2 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 12, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf))
        self.decoder_1 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 14, output_nc, kernel_size=4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, input):
        x1 = self.coder_1(input)
        x2 = self.coder_2(x1)
        x3 = self.coder_3(x2)
        x4 = self.coder_4(x3)
        x5 = self.coder_5(x4)
        x6 = self.coder_6(x5)
        x7 = self.coder_7(x6)
        y7 = self.innermost_8(x7)
        y6 = self.decoder_7(torch.cat([x7, y7], 1))
        y5 = self.decoder_6(torch.cat([x6, y6], 1))
        y4 = self.decoder_5(torch.cat([x5, y5], 1))
        y4to2 = F.interpolate(y4, scale_factor=4, mode='bilinear', align_corners=True)
        y4to1 = F.interpolate(y4, scale_factor=8, mode='bilinear', align_corners=True)
        y3 = self.decoder_4(torch.cat([x4, y4], 1))
        y3to1 = F.interpolate(y3, scale_factor=4, mode='bilinear', align_corners=True)
        y2 = self.decoder_3(torch.cat([x3, y3], 1))
        y1 = self.decoder_2(torch.cat([x2, y4to2, y2], 1))
        output = self.decoder_1(torch.cat([x1, y4to1, y3to1, y1], 1))
        return output, y1, y2, y3, y4

class dsuGeneratorRGB2NIR(nn.Module):
    def __init__(self, input_nc, output_nc, norm_G, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, use_attention=True):
        super(dsuGeneratorRGB2NIR, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.image_size = [128, 64, 32, 16]
        self.coder_1 = nn.Sequential(nn.Conv2d(1, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), MambaIR(img_size=self.image_size[0], embed_dim=ngf))
        self.coder_2 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), MambaIR(img_size=self.image_size[1], embed_dim=ngf * 2), norm_layer(ngf * 2))
        self.coder_3 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), MambaIR(img_size=self.image_size[2], embed_dim=ngf * 4), norm_layer(ngf * 4))
        self.coder_4 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.coder_5 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.coder_6 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.coder_7 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.innermost_8 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_7 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_6 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_5 = decoder(ngf * 16, ngf * 8, norm_G)
        self.decoder_4 = decoder(ngf * 16, ngf * 4, norm_G)
        self.decoder_3 = decoder(ngf * 8, ngf * 2, norm_G, self.image_size[2], Mamba_use=False)
        self.decoder_2 = decoder(ngf * 12, ngf, norm_G, self.image_size[1], Mamba_use=False)
        self.block = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 14, ngf, kernel_size=4, stride=2, padding=1))
        self.fusion = SPADEResnetBlock(3, 3, 1, norm_G)
        self.cross_block = CrissCrossAttention(ngf, 3)
        self.fin_out = nn.Sequential(nn.Conv2d(ngf, 3, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, input, hsv, f1, f2, f3, f4):
        x1 = self.coder_1(input)
        x2 = self.coder_2(x1)
        x3 = self.coder_3(x2)
        x4 = self.coder_4(x3)
        x5 = self.coder_5(x4)
        x6 = self.coder_6(x5)
        x7 = self.coder_7(x6)
        y7 = self.innermost_8(x7)
        y6 = self.decoder_7(torch.cat([x7, y7], 1))
        y5 = self.decoder_6(torch.cat([x6, y6], 1))
        y4 = self.decoder_5(torch.cat([x5, y5], 1), f4)
        y4to2 = F.interpolate(y4, scale_factor=4, mode='bilinear', align_corners=True)
        y4to1 = F.interpolate(y4, scale_factor=8, mode='bilinear', align_corners=True)
        y3 = self.decoder_4(torch.cat([x4, y4], 1), f3)
        y3to1 = F.interpolate(y3, scale_factor=4, mode='bilinear', align_corners=True)
        y2 = self.decoder_3(torch.cat([x3, y3], 1), f2)
        y1 = self.decoder_2(torch.cat([x2, y4to2, y2], 1), f1)
        segmap = self.apply_laplacian(input)[:, None]
        feature = self.block(torch.cat([x1, y4to1, y3to1, y1], 1))
        seg_hsv = self.fusion(hsv, segmap)
        seg_hsv = self.cross_block(feature, seg_hsv)
        output = self.fin_out(seg_hsv)
        return output

    def apply_laplacian(self, input):
        nir_image = (input.detach().cpu().numpy()[:, 0] * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(nir_image, cv2.CV_64F)
        epsilon = 1e-8
        laplacian = (laplacian - np.min(laplacian)) / ((np.max(laplacian) - np.min(laplacian)) + epsilon)
        laplacian_tensor = torch.from_numpy(laplacian).to(input.device)
        return laplacian_tensor

class dsuGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(dsuGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.coder_1 = nn.Sequential(nn.Conv2d(3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias))
        self.coder_2 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2))
        self.coder_3 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 4))
        self.coder_4 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.coder_5 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.coder_6 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.coder_7 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.innermost_8 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_7 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_6 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_5 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8))
        self.decoder_4 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 4))
        self.decoder_3 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2))
        self.decoder_2 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 12, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf))
        self.decoder_1 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 14, output_nc, kernel_size=4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, input):
        x1 = self.coder_1(input)
        x2 = self.coder_2(x1)
        x3 = self.coder_3(x2)
        x4 = self.coder_4(x3)
        x5 = self.coder_5(x4)
        x6 = self.coder_6(x5)
        x7 = self.coder_7(x6)
        y7 = self.innermost_8(x7)
        y6 = self.decoder_7(torch.cat([x7, y7], 1))
        y5 = self.decoder_6(torch.cat([x6, y6], 1))
        y4 = self.decoder_5(torch.cat([x5, y5], 1))
        y4to2 = F.interpolate(y4, scale_factor=4, mode='bilinear', align_corners=True)
        y4to1 = F.interpolate(y4, scale_factor=8, mode='bilinear', align_corners=True)
        y3 = self.decoder_4(torch.cat([x4, y4], 1))
        y3to1 = F.interpolate(y3, scale_factor=4, mode='bilinear', align_corners=True)
        y2 = self.decoder_3(torch.cat([x3, y3], 1))
        y1 = self.decoder_2(torch.cat([x2, y4to2, y2], 1))
        output = self.decoder_1(torch.cat([x1, y4to1, y3to1, y1], 1))
        return output

class all_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        self.netG_A = dsuGeneratorRGB2NIR(input_nc, output_nc, 'spectralinstance', ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.netG_C = UNetGenerator(ngf // 2)

    def forward(self, nir_img, nir_hsv_img):
        fake_B_hsv, f1, f2, f3, f4 = self.netG_C(nir_img, nir_hsv_img)
        fake_B = self.netG_A(nir_img, fake_B_hsv, f1, f2, f3, f4)
        return fake_B_hsv, fake_B

class all_Generator_Small(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        self.netG_A = dsuGeneratorRGB2NIR_Small(input_nc, output_nc, 'spectralinstance', ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.netG_C = UNetGenerator(ngf // 2)

    def forward(self, nir_img, nir_hsv_img):
        fake_B_hsv, f1, f2, f3, f4 = self.netG_C(nir_img, nir_hsv_img)
        fake_B = self.netG_A(nir_img, fake_B_hsv, f1, f2, f3, f4)
        return fake_B_hsv, fake_B

class dsuGeneratorRGB2NIR_Small(nn.Module):
    def __init__(self, input_nc, output_nc, norm_G, ngf=32, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, use_attention=True):
        super(dsuGeneratorRGB2NIR_Small, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.image_size = [128, 64, 32, 16]
        self.coder_1 = nn.Sequential(nn.Conv2d(1, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), MambaIR(img_size=self.image_size[0], embed_dim=ngf))
        self.coder_2 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), MambaIR(img_size=self.image_size[1], embed_dim=ngf * 2), norm_layer(ngf * 2))
        self.coder_3 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 4))
        self.decoder_3 = decoder(ngf * 4, ngf * 2, norm_G, self.image_size[2], Mamba_use=False)
        self.decoder_2 = decoder(ngf * 4, ngf, norm_G, self.image_size[1], Mamba_use=False)
        self.decoder_1 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1), nn.Sigmoid())
        self.block = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1))  # Fixed to ngf * 2 (64 channels)
        self.fusion = SPADEResnetBlock(3, 3, 1, norm_G)
        self.cross_block = CrissCrossAttention(ngf, 3)
        self.fin_out = nn.Sequential(nn.Conv2d(ngf, 3, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, input, hsv, f1, f2, f3, f4):
        x1 = self.coder_1(input)
        x2 = self.coder_2(x1)
        x3 = self.coder_3(x2)
        y2 = self.decoder_3(x3, f2)
        y1 = self.decoder_2(torch.cat([x2, y2], 1), f1)
        segmap = self.apply_laplacian(input)[:, None]
        feature = self.block(torch.cat([x1, y1], 1))
        seg_hsv = self.fusion(hsv, segmap)
        seg_hsv = self.cross_block(feature, seg_hsv)
        output = self.fin_out(seg_hsv)
        return output

    def apply_laplacian(self, input):
        nir_image = (input.detach().cpu().numpy()[:, 0] * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(nir_image, cv2.CV_64F)
        epsilon = 1e-8
        laplacian = (laplacian - np.min(laplacian)) / ((np.max(laplacian) - np.min(laplacian)) + epsilon)
        laplacian_tensor = torch.from_numpy(laplacian).to(input.device)
        return laplacian_tensor

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        norm_layer = SpectralNorm
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

if __name__ == '__main__':
    model = all_Generator_Small(3, 3)
    x_gray = torch.randn(1, 1, 256, 256)
    x_hsv = torch.randn(1, 3, 256, 256)
    fake_hsv, fake_rgb = model(x_gray, x_hsv)
    print(fake_rgb.shape)