# Script para gerar um relatório PDF com resultados e gráficos do modelo de difusão

from fpdf import FPDF
import pandas as pd
import os

# Caminhos esperados
dir_resultados = "resultados_sample"
csv_path = os.path.join(dir_resultados, "metricas_todas.csv")
plot_path = os.path.join(dir_resultados, "metricas_boxplot.png")
pdf_path = os.path.join(dir_resultados, "relatorio_resultados.pdf")

# Carrega o CSV com as métricas
df = pd.read_csv(csv_path)

# Cria o objeto PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Título
pdf.set_font("Arial", style="B", size=16)
pdf.cell(0, 10, "Relatório de Geração com Modelo de Difusão", ln=True, align='C')

# Texto introdutório
pdf.set_font("Arial", size=12)
pdf.ln(10)
pdf.multi_cell(0, 8, "Este relatório apresenta os resultados obtidos com o modelo de difusão para diferentes quantidades de etapas de amostragem. São avaliadas as métricas SSIM e MSE para cada uma das {} amostras geradas por etapa, além de exemplos visuais do processo de geração.".format(df["Amostra"].nunique()))

# Adiciona a imagem do boxplot
if os.path.exists(plot_path):
    pdf.ln(10)
    pdf.image(plot_path, w=180)

# Métricas agregadas
pdf.ln(10)
pdf.set_font("Arial", style="B", size=12)
pdf.cell(0, 10, "Estatísticas das Métricas", ln=True)

df_summary = df.groupby("Etapas").agg({"MSE": ["mean", "std"], "SSIM": ["mean", "std"]}).round(4)
summary_txt = df_summary.to_string()

pdf.set_font("Courier", size=10)
pdf.multi_cell(0, 5, summary_txt)

# Exemplos visuais para todas as amostras por etapa
pdf.set_font("Arial", style="B", size=12)
pdf.ln(10)
pdf.cell(0, 10, "Exemplos Visuais por Etapas e Amostras", ln=True)

for etapa in sorted(df["Etapas"].unique()):
    for idx in sorted(df[df["Etapas"] == etapa]["Amostra"].unique()):
        pasta = os.path.join(dir_resultados, f"sample{int(etapa)}_idx{int(idx)}")
        img_path = os.path.join(pasta, "resultado.png")
        if os.path.exists(img_path):
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 8, f"Etapas: {int(etapa)} | Amostra: {int(idx)}", ln=True)
            pdf.image(img_path, w=64)

# Salva o PDF
pdf.output(pdf_path)
print(f"Relatório salvo em: {pdf_path}")
