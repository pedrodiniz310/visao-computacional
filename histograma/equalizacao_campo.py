"""
Visão Computacional — Equalização de Histograma
Aplica equalização de histograma na imagem campo.jpg
para melhorar o contraste de uma foto com iluminação ruim.
"""

import os
import cv2
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/campo.jpg'))

# ── Carregamento em escala de cinza ──────────────────────────────────────────────
# cv2.IMREAD_GRAYSCALE (=0): carrega em 1 canal (8 bits, 0-255)
# Equalização de histograma opera apenas em imagens de 1 canal
imagem = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Equalização global e CLAHE ──────────────────────────────────────────────────
# cv2.equalizeHist(): redistribui os pixels para cobrir [0-255] uniformemente
# Problema: trata a imagem INTEIRA como uma só → pode sobre-amplificar regiões
# já claras e criar artefatos em imagens com contraste não-uniforme
equalizada_global = cv2.equalizeHist(imagem)

# cv2.createCLAHE(clipLimit, tileGridSize):
#   clipLimit=2.0: limita a amplificação de contraste para evitar ruído
#     (valores altos = mais contraste mas mais granulado)
#   tileGridSize=(8,8): divide a imagem em 64 tiles (8×8)
#     cada tile é equalizado de forma INDEPENDENTE → contraste LOCAL
clahe            = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalizada_clahe = clahe.apply(imagem)  # aplica o CLAHE na imagem

# ── Histogramas ───────────────────────────────────────────────────────────────
# cv2.calcHist([imgs], [canais], máscara, [nBins], [range])
#   [0] = canal 0 (único canal na imagem em escala de cinza)
#   None = sem máscara (processa toda a imagem)
#   [256] = 256 bins (um por nível de intensidade)
#   [0, 256] = faixa de valores analisada
hist_orig    = cv2.calcHist([imagem],           [0], None, [256], [0, 256])
hist_global  = cv2.calcHist([equalizada_global],[0], None, [256], [0, 256])
hist_clahe   = cv2.calcHist([equalizada_clahe], [0], None, [256], [0, 256])

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Equalização de Histograma — Campo', fontsize=13)

axes[0, 0].imshow(imagem,           cmap='gray'); axes[0, 0].set_title('Original');              axes[0, 0].axis('off')
axes[0, 1].imshow(equalizada_global, cmap='gray'); axes[0, 1].set_title('Global (equalizeHist)'); axes[0, 1].axis('off')
axes[0, 2].imshow(equalizada_clahe,  cmap='gray'); axes[0, 2].set_title('CLAHE (local)');         axes[0, 2].axis('off')

axes[1, 0].plot(hist_orig);   axes[1, 0].set_xlim([0, 256]); axes[1, 0].set_title('Histograma Original')
axes[1, 1].plot(hist_global); axes[1, 1].set_xlim([0, 256]); axes[1, 1].set_title('Histograma Global')
axes[1, 2].plot(hist_clahe);  axes[1, 2].set_xlim([0, 256]); axes[1, 2].set_title('Histograma CLAHE')

plt.tight_layout()
plt.show()