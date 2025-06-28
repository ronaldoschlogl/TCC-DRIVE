import matplotlib.pyplot as plt
import re

# Caminho para o arquivo contendo os registros da função de perda
caminho_arquivo = "loss.txt"

# Listas para armazenar os valores extraídos
iteracoes = []
perda_total = []
perda_simplificada = []
vlb = []

# Expressão regular para extrair os valores de cada linha do arquivo
padrao = r"Itr: (\d+), Loss: ([\d.]+), L_s: ([\d.]+), L_vlb: ([\d.]+)"

# Leitura do arquivo linha por linha
with open(caminho_arquivo, "r") as arquivo:
    for linha in arquivo:
        match = re.match(padrao, linha)
        if match:
            iteracoes.append(int(match.group(1)))
            perda_total.append(float(match.group(2)))
            perda_simplificada.append(float(match.group(3)))
            vlb.append(float(match.group(4)))

# Geração do gráfico com Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(iteracoes, perda_total, label="Loss total", linewidth=2)
plt.plot(iteracoes, perda_simplificada, label="L_s (simplified loss)", linestyle="--")
plt.plot(iteracoes, vlb, label="L_vlb (variational term)", linestyle=":")
plt.xlabel("Iterações")
plt.ylabel("Valor da perda")
plt.title("Evolução da Função de Perda durante o Treinamento")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("evolucao_perda.png")  # Salva o gráfico como imagem
plt.show()
