"""
Visão Computacional — Aula 03
Conversão entre espaços de cor: BGR → Cinza, BGR → HSV.
Separação e visualização de canais.
"""

import os
import cv2
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/frutas.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem_bgr = cv2.imread(PATH)
if imagem_bgr is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Conversões ────────────────────────────────────────────────────────────────
# BGR → RGB: inverte ordem dos canais B e R para exibir com matplotlib
imagem_rgb  = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
# BGR → Cinza: Y = 0.114*B + 0.587*G + 0.299*R (fórmula de luminância ITU-R BT.601)
imagem_gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
# BGR → HSV: H(matiz)=0-179, S(saturação)=0-255, V(brilho)=0-255
# H é independente de brilho → ideal para segmentar objetos por cor
imagem_hsv  = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

# cv2.split(): separa os 3 canais em 3 arrays independentes de 1 canal cada
h, s, v = cv2.split(imagem_hsv)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Espaços de Cor — BGR, Cinza e HSV', fontsize=13)

axes[0, 0].imshow(imagem_rgb);       axes[0, 0].set_title('RGB Original');         axes[0, 0].axis('off')
axes[0, 1].imshow(imagem_gray, cmap='gray'); axes[0, 1].set_title('Escala de Cinza'); axes[0, 1].axis('off')
axes[0, 2].imshow(cv2.cvtColor(imagem_hsv, cv2.COLOR_HSV2RGB)); axes[0, 2].set_title('HSV → RGB'); axes[0, 2].axis('off')

axes[1, 0].imshow(h, cmap='hsv');    axes[1, 0].set_title('Canal H (Matiz)');      axes[1, 0].axis('off')
axes[1, 1].imshow(s, cmap='gray');   axes[1, 1].set_title('Canal S (Saturação)');  axes[1, 1].axis('off')
axes[1, 2].imshow(v, cmap='gray');   axes[1, 2].set_title('Canal V (Valor/Brilho)'); axes[1, 2].axis('off')

plt.tight_layout()
plt.show()