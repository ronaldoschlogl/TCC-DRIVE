# Script de treinamento para modelo de difusão usando UNet, EMA e amostragem espacial

import os
import glob
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image
from models import UNet
from Dataset import Dataset_celeba
from gaussian_diffusion import GaussianDiffusion, SpaceSampling
from ema_pytorch import EMA
import argparse

# Função principal de treinamento
def treinamento(modelo_salvo):
    # Hiperparâmetros
    num_sample = 250                  # Etapas de amostragem para visualização
    epoch = 1000                      # Número de épocas de treinamento
    batchsize = 8                     # Tamanho do lote
    ema_decay = 0.9999                # Decaimento do EMA
    ema_update_every = 10            # Frequência de atualização do EMA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 64
    noise_type = "cos"
    timesteps = 1000
    model_channel = 192
    channel_mult = [1, 2, 3, 4]
    attention_resolution = [8, 16, 32]
    dropout = 0.1
    learning_rate = 1e-4

    # Carrega dataset CelebA
    dataset = Dataset_celeba(path="./img_align_celeba/", image_size=image_size)

    # Instancia o modelo UNet
    model = UNet(image_size=image_size,
                 in_channel=3,
                 out_channel=6,
                 model_channel=model_channel,
                 channel_mult=channel_mult,
                 attention_resolutions=attention_resolution,
                 dropout=dropout,
                 device=device).to(device)
    model.train()

    # Otimizador
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batchsize,
                            shuffle=True, num_workers=8, drop_last=True)

    # Instancia o processo de difusão
    gau_diff = GaussianDiffusion(noise_type=noise_type,
                                  timesteps=timesteps,
                                  device=device)

    # Instancia o sampler para geração de imagens
    space_sample = SpaceSampling(noise_type=noise_type,
                                  timesteps=timesteps,
                                  num_sample=num_sample,
                                  device=device)

    # Inicializa EMA
    ema_model = EMA(model, beta=ema_decay, update_every=ema_update_every).to(device)

    # Se existir um modelo salvo, carrega os pesos
    if modelo_salvo is not None and os.path.exists(modelo_salvo):
        checkpoint = torch.load(modelo_salvo)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        del checkpoint

    # Criação de diretórios de saída
    os.makedirs("./resultados", exist_ok=True)
    os.makedirs("./modelos_salvos", exist_ok=True)

    # Arquivo para registrar a perda
    with open("loss.txt", "w") as f:
        total_itr = 0

        for ep in range(epoch):
            for itr, (batch_x0, labels) in enumerate(dataloader):
                print(f"Iteração: {itr}")
                batch_x0, labels = batch_x0.to(device), labels.to(device)
                optimizer.zero_grad()

                # Amostragem aleatória do tempo t
                t = torch.randint(0, timesteps, [batchsize]).to(device)

                # Calcula a função de perda
                loss, L_s, L_vlb = gau_diff.p_losses(model, batch_x0, t, labels)
                loss.backward()
                optimizer.step()
                ema_model.update()

                # Log e gravação da perda
                print(f"Itr: {total_itr}, Loss: {loss.item()}, L_s: {L_s.item()}, L_vlb: {L_vlb.item()}")
                f.write(f"Itr: {total_itr}, Loss: {loss.item()}, L_s: {L_s.item()}, L_vlb: {L_vlb.item()}\n")
                f.flush()

                # A cada 500 iterações: gera uma imagem com modelo EMA
                if total_itr % 500 == 0:
                    ema_model.eval()
                    with torch.no_grad():
                        pred_x_0 = space_sample.p_fast_sample_loop(ema_model, y=None, image_size=image_size)
                        pred_x_0 = ((pred_x_0.permute(0, 2, 3, 1)).clamp(-1, 1) + 1) * 127.5
                        pred_img = pred_x_0.cpu().numpy()[0].astype(np.uint8)
                        Image.fromarray(pred_img).save(f"./resultados/{total_itr}.png")

                # A cada 1000 iterações: salva o modelo
                if total_itr % 1000 == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema_model.state_dict()
                    }
                    torch.save(checkpoint, f"./modelos_salvos/model_{total_itr}.pth")

                total_itr += 1
            print(f"Final da época {ep + 1}")

# Execução principal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    files = glob.glob('./modelos_salvos/*.pth')
    latest_file = max(files, key=os.path.getmtime) if files else None
    parser.add_argument("--modelo_salvo", type=str, default=latest_file)
    args = parser.parse_args()

    treinamento(args.modelo_salvo)
