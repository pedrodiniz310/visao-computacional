"""
Visão Computacional — Quantização de Tons de Cinza
Reduz a profundidade de bits da imagem Chaplin de 8-bit
para 4, 2 e 1 bit, demonstrando o efeito da quantização.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/chaplin.jpg'))

# ── Carregamento em escala de cinza ──────────────────────────────────────────────
imagem = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Quantização: reduzir profundidade de bits ───────────────────────────────────
def quantizar(img, bits):
    """Reduz a imagem para 'bits' bits de profundidade.

    Método: divide os níveis em grupos e mapeia cada grupo para o menor valor.
    Ex: 4 bits = 16 níveis → passo = 256/16 = 16
        pixeis 0-15 → 0; 16-31 → 16; 32-47 → 32; ...; 240-255 → 240
    """
    niveis = 2 ** bits      # número de tons distintos (4 bits = 16 tons)
    passo  = 256 // niveis  # largura de cada faixa de intensidade
    # Divide cada pixel pelo passo (trunca para o menor valor do grupo)
    # Multiplica de volta para restaurar a escala original [0-255]
    # .astype(np.uint8) garante que os valores fiquem em 8 bits
    return (img // passo * passo).astype(np.uint8)

q4 = quantizar(imagem, 4)   # 4 bits  = 16 níveis de cinza  (perda leve)
q2 = quantizar(imagem, 2)   # 2 bits  =  4 níveis de cinza  (perda moderada)
q1 = quantizar(imagem, 1)   # 1 bit   =  2 níveis (binária: preto ou branco)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Quantização — Redução da Profundidade de Bits', fontsize=13)

axes[0].imshow(imagem, cmap='gray', vmin=0, vmax=255); axes[0].set_title('Original (8 bits\n256 níveis)'); axes[0].axis('off')
axes[1].imshow(q4,     cmap='gray', vmin=0, vmax=255); axes[1].set_title('4 bits\n16 níveis');             axes[1].axis('off')
axes[2].imshow(q2,     cmap='gray', vmin=0, vmax=255); axes[2].set_title('2 bits\n4 níveis');              axes[2].axis('off')
axes[3].imshow(q1,     cmap='gray', vmin=0, vmax=255); axes[3].set_title('1 bit\n2 níveis (binária)');     axes[3].axis('off')

plt.tight_layout()
plt.show()