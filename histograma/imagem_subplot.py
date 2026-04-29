"""
Visão Computacional — Imagem com Subplot de Histograma
Exibe a imagem frutas.png ao lado do histograma
dos seus três canais de cor (B, G, R).
"""

import os
import cv2
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/frutas.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem_bgr = cv2.imread(PATH)
if imagem_bgr is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# cv2.COLOR_BGR2RGB: inverte a ordem dos canais
# OpenCV lê imagens como BGR (Blue-Green-Red), mas matplotlib exibe como RGB
# Sem essa conversão, as cores apareceriam invertidas (vermelho e azul trocados)
imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

# ── Cálculo dos histogramas por canal ────────────────────────────────────────────
cores  = ('b', 'g', 'r')                           # cores do matplotlib para cada canal
nomes  = ('Azul (B)', 'Verde (G)', 'Vermelho (R)')  # nomes para a legenda
# Calcula 1 histograma por canal: [i] seleciona o canal 0 (B), 1 (G) ou 2 (R)
# A imagem é passada em BGR pois os índices [0,1,2] = [B,G,R] no OpenCV
hists  = [cv2.calcHist([imagem_bgr], [i], None, [256], [0, 256]) for i in range(3)]

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Imagem e Histograma dos Canais BGR — Frutas', fontsize=13)

axes[0].imshow(imagem_rgb)
axes[0].set_title('Imagem Original'); axes[0].axis('off')

for hist, cor, nome in zip(hists, cores, nomes):
    axes[1].plot(hist, color=cor, label=nome, alpha=0.8)

axes[1].set_xlim([0, 256])
axes[1].set_xlabel('Intensidade')
axes[1].set_ylabel('Número de pixels')
axes[1].set_title('Histograma por Canal de Cor')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()