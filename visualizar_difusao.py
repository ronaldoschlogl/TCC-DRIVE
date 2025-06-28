import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

# Caminho para a base de imagens
image_dir = "./img_align_celeba/"  # ajuste se necessário
image_size = 128
timesteps = 5  # número de etapas de difusão

# Carregar uma imagem da base CelebA
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".jpg")]
image = Image.open(image_paths[0]).convert("RGB")

# Transformações: redimensionamento, normalização para [-1,1]
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
x_0 = transform(image).unsqueeze(0)  # (1, 3, H, W)

# Simular o processo de difusão direta
alphas = torch.linspace(1.0, 0.01, steps=timesteps)  # decaimento suave
noisy_images = [x_0]
for alpha in alphas[1:]:
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha) * x_0 + torch.sqrt(1 - alpha) * noise
    noisy_images.append(x_t)

# Salvar visualização em grade
output_dir = "./resultado_difusao_direta"
os.makedirs(output_dir, exist_ok=True)
grid_path = os.path.join(output_dir, "diffusion_forward_process.png")
save_image(torch.cat(noisy_images, dim=0), grid_path, nrow=timesteps, normalize=True, value_range=(-1, 1))

print(f"Imagem do processo de difusão direta salva em: {grid_path}")
