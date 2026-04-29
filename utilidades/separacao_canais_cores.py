"""
Visão Computacional — Separação de Canais de Cor
Separa os canais B, G, R da imagem e exibe cada um individualmente.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/cores.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem_bgr = cv2.imread(PATH)
if imagem_bgr is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

# ── Separação dos canais BGR ──────────────────────────────────────────────────
# cv2.split(): separa a imagem BGR em 3 arrays de 1 canal cada (shape: H×W)
# b=azul (canal 0), g=verde (canal 1), r=vermelho (canal 2)
b, g, r = cv2.split(imagem_bgr)

# Reconstruir cada canal como imagem colorida (para visualização intuitiva)
# np.zeros_like(b): array de zeros com mesmo shape e dtype que b (8 bits)
# cv2.merge([b, zeros, zeros]): imagem BGR onde apenas o azul tem valor real
# → pixels aparecem na cor azul com intensidade proporcional ao valor do canal B
zeros   = np.zeros_like(b)
canal_b = cv2.merge([b, zeros, zeros])   # somente azul
canal_g = cv2.merge([zeros, g, zeros])   # somente verde
canal_r = cv2.merge([zeros, zeros, r])   # somente vermelho

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Separação dos Canais de Cor (BGR)', fontsize=13)

axes[0, 0].imshow(imagem_rgb)
axes[0, 0].set_title('Imagem Original (RGB)'); axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(canal_b, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Canal B — Azul'); axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(canal_g, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Canal G — Verde'); axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(canal_r, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Canal R — Vermelho'); axes[1, 1].axis('off')

plt.tight_layout()
plt.show()