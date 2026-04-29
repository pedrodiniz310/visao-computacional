"""
Visão Computacional — Histograma de Imagem em Preto e Branco
Plota o histograma da imagem folha_pb.png em escala de cinza,
mostrando a distribuição dos tons de intensidade.
"""

import os
import cv2
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/folha_pb.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Cálculo do histograma ─────────────────────────────────────────────────────
# cv2.calcHist([imgs], [canais], máscara, [nBins], [range])
#   [imagem]: lista com 1 imagem (obrigatório passar como lista)
#   [0]: índice do canal a analisar (0 = único canal da imagem em cinza)
#   None: sem máscara — analisa TODOS os pixels
#   [256]: número de bins (colunas do histograma) → 1 bin = 1 nível de intensidade
#   [0, 256]: intervalo [mínimo, máximo] dos valores a contar
# Retorno: array de shape (256, 1) com a contagem de pixels por nível
hist = cv2.calcHist([imagem], [0], None, [256], [0, 256])

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Histograma — Folha em Preto e Branco', fontsize=13)

ax1.imshow(imagem, cmap='gray')
ax1.set_title('Imagem Original (Cinza)'); ax1.axis('off')

ax2.plot(hist, color='black')
ax2.fill_between(range(256), hist.flatten(), alpha=0.4, color='gray')
ax2.set_xlim([0, 256])
ax2.set_xlabel('Intensidade (0=preto, 255=branco)')
ax2.set_ylabel('Número de pixels')
ax2.set_title('Histograma de Tons de Cinza')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()