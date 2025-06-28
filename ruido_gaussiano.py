import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Cria a figura
fig, ax = plt.subplots(figsize=(12, 2.5))

# Cria os estados da cadeia de Markov (x0 até xT)
T = 6
states = [f"x{i}" for i in range(T + 1)]
positions = range(len(states))

# Adiciona os círculos representando os estados
for pos, label in zip(positions, states):
    circle = patches.Circle((pos * 1.5, 0), 0.3, edgecolor='black', facecolor='lightgray')
    ax.add_patch(circle)
    ax.text(pos * 1.5, 0, label, ha='center', va='center')

# Adiciona setas direcionais
for i in range(T):
    ax.annotate("",
                xy=(i * 1.5 + 0.3, 0), 
                xytext=((i + 1) * 1.5 - 0.3, 0),
                arrowprops=dict(arrowstyle="->", lw=1.5))

# Adiciona rótulos de transição
for i in range(T):
    ax.text(i * 1.5 + 0.75, 0.3, f"q(x{i+1}|x{i})", ha='center')

# Limites e ajustes do gráfico
ax.set_xlim(-0.5, (T + 1) * 1.5 - 1)
ax.set_ylim(-1, 1)
ax.axis('off')
plt.title("Processo de Difusão: Adição Progressiva de Ruído Gaussiano", fontsize=13)
plt.tight_layout()
plt.show()
