import os
import torch
from models import UNet
from PIL import Image
import numpy as np
from gaussian_diffusion import SpaceSampling
from ema_pytorch import EMA
import argparse
import glob
import matplotlib.pyplot as plt


def inference(modelo):
    device = "cuda"
    num_sample = 250
    image_size = 64
    noise_type = "cos"
    timesteps = 1000
    model_channel = 192
    num_heads = 4
    num_res_block = 3
    num_head_channel = 64
    channel_mult = [1, 2, 3, 4]
    attention_resolution = [8, 16, 32]
    dropout = 0.1

    model_diffusion = UNet(image_size=image_size,
                           in_channel=3,
                           out_channel=6,    # [pred_noise, pred_log_var_frac]
                           num_class=None,
                           model_channel=model_channel,
                           channel_mult=channel_mult,
                           attention_resolutions=attention_resolution,
                           num_res_block=num_res_block,
                           num_head_channel=num_head_channel,
                           num_heads=num_heads,
                           dropout=dropout,
                           num_groups=32,
                           device=device)

    model_diffusion.eval()
    model_diffusion = EMA(model_diffusion)
    model_diffusion = model_diffusion.to(device)
    model_diffusion.load_state_dict(torch.load(modelo)["ema"])
    ss = SpaceSampling(noise_type=noise_type, timesteps=timesteps,
                       num_sample=num_sample, device=device)
    res = []
    for i in range(5):
        pred_x_0 = ss.p_fast_sample_loop(
            model_diffusion, image_size=image_size)
        pred_x_0_ = ((pred_x_0.permute(0, 2, 3, 1)).clamp(-1, 1) + 1) * 127.5
        pred_x_0_ = pred_x_0_.detach().cpu().numpy()[0]
        res.append(pred_x_0_)
    res = np.concatenate(res, axis=0)
    Image.fromarray(np.uint8(res)).save(f"resultado.png")

    # plt.figure(figsize=(3, 3))
    # plt.imshow(Image.fromarray(np.uint8(res)))
    # plt.axis('off')
    # plt.show()


if __name__ == "__main__":
    files = glob.glob('./modelos_salvos/*')
    latest_file = max(files, key=os.path.basename)
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=str, default=latest_file)
    args = parser.parse_args()
    inference(args.modelo)
