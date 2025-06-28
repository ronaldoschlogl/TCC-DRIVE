# Script adaptado para medir SSIM, MSE e tempo de geração de cada amostra

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from models import UNet
from gaussian_diffusion import SpaceSampling
from ema_pytorch import EMA
from tqdm import tqdm
import imageio
import time  # Para medir o tempo de execução

# Parâmetros principais
modelo_caminho = "./modelos_salvos/model_20000.pth"
pasta_resultados = "resultados_sample"
os.makedirs(pasta_resultados, exist_ok=True)

etapas_amostragem = [75, 125, 250, 500, 750, 1000]  # Etapas de amostragem para comparação
num_amostras = 1000
image_size = 64
timesteps = 1000
noise_type = "cos"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Gera ruído base fixo para consistência
torch.manual_seed(42)
ruido_base_global = torch.randn(1, 3, image_size, image_size, device=device)

# Parâmetros do modelo UNet
model_params = {
    "image_size": image_size,
    "in_channel": 3,
    "out_channel": 6,
    "num_class": None,
    "model_channel": 192,
    "channel_mult": [1, 2, 3, 4],
    "attention_resolutions": [8, 16, 32],
    "num_res_block": 3,
    "num_head_channel": 64,
    "num_heads": 4,
    "dropout": 0.1,
    "num_groups": 32,
    "device": device
}

# Inicializa e carrega modelo com EMA
model = UNet(**model_params).eval()
ema_model = EMA(model).to(device)
ema_model.load_state_dict(torch.load(modelo_caminho)["ema"])

# Converte tensor para imagem PIL
def tensor_para_imagem(tensor):
    imagem = ((tensor.permute(0, 2, 3, 1)).clamp(-1, 1) + 1) * 127.5
    return imagem.detach().cpu().numpy()[0].astype(np.uint8)

# Lista para armazenar métricas
metricas_total = []

# Loop para diferentes números de etapas
for ns in etapas_amostragem:
    print(f"Gerando {num_amostras} amostras para {ns} etapas")
    sampler = SpaceSampling(
        noise_type=noise_type,
        timesteps=timesteps,
        num_sample=ns,
        device=device
    )

    for i in range(num_amostras):
        nome_base = f"sample{ns}_idx{i}"
        x_t = ruido_base_global.clone()
        ruido_np = tensor_para_imagem(x_t)

        frames = []
        start_time = time.time()  # Início da contagem de tempo

        for t in tqdm(range(ns - 1, -1, -1), desc=f"Sampling t={ns}"):
            t_tensor = torch.tensor([t], dtype=torch.int64, device=device)
            x_t = sampler.p_fast_sample(ema_model, x_t, t_tensor)
            frame_np = tensor_para_imagem(x_t)
            frames.append(Image.fromarray(frame_np))

        end_time = time.time()
        tempo_execucao = end_time - start_time

        # Salva imagem final e GIF
        Image.fromarray(frame_np).save(os.path.join(pasta_resultados, f"{nome_base}_resultado.png"))
        # frames[0].save(os.path.join(pasta_resultados, f"{nome_base}_evolucao.gif"), save_all=True, append_images=frames[1:], duration=100, loop=0)

        # Calcula métricas
        mse_val = mean_squared_error(ruido_np.flatten(), frame_np.flatten())
        ssim_val = ssim(ruido_np, frame_np, channel_axis=2, data_range=255)
        metricas_total.append((ns, i, mse_val, ssim_val, tempo_execucao))

        # Adiciona ao CSV
        csv_path = os.path.join(pasta_resultados, "metricas.csv")
        escrever_cabecalho = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if escrever_cabecalho:
                f.write("Etapas,Amostra,MSE,SSIM,Tempo_execucao_segundos\n")
            f.write(f"{ns},{i},{mse_val},{ssim_val},{tempo_execucao:.4f}\n")

# Salva todas as métricas em um arquivo final
df = pd.DataFrame(metricas_total, columns=["Etapas", "Amostra", "MSE", "SSIM", "Tempo_execucao_segundos"])
df.to_csv(os.path.join(pasta_resultados, "metricas_todas.csv"), index=False)

# Gera boxplot de SSIM
plt.figure(figsize=(10, 6))
df.boxplot(column="SSIM", by="Etapas")
plt.title("Distribuição do SSIM por Etapas")
plt.suptitle("")
plt.xlabel("Etapas de Amostragem")
plt.ylabel("SSIM")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pasta_resultados, "metricas_boxplot.png"))
plt.close()

if __name__ == "__main__":
    print("Amostragem completa.")

