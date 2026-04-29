"""
Visão Computacional — Clarear e Escurecer Imagem
Demonstra operações ponto a ponto para clarear (+80)
e escurecer (-80) a imagem frutas.png.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/frutas.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem_bgr = cv2.imread(PATH)
if imagem_bgr is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

# ── Operações ponto a ponto ───────────────────────────────────────────────────
# cv2.add e cv2.subtract respeitam SATURAÇÃO (0-255), sem overflow!
# Sem saturação, somar 255 + 80 daria 335 → wrap-around para 79 (resultado errado)
# np.full(shape, valor, dtype) cria array com todos os pixels = valor
# A subtração escurece: cada pixel fica com (valor - 80), mínimo = 0
escura = cv2.subtract(imagem_bgr, np.full(imagem_bgr.shape, 80, dtype=np.uint8))
# A adição clareia: cada pixel fica com (valor + 80), máximo = 255
clara  = cv2.add(     imagem_bgr, np.full(imagem_bgr.shape, 80, dtype=np.uint8))

# Converter para RGB para exibir corretamente com matplotlib
escura_rgb = cv2.cvtColor(escura, cv2.COLOR_BGR2RGB)
clara_rgb  = cv2.cvtColor(clara,  cv2.COLOR_BGR2RGB)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Operações Ponto a Ponto — Clarear e Escurecer', fontsize=13)

axes[0].imshow(escura_rgb); axes[0].set_title('Escurecida  (−80)'); axes[0].axis('off')
axes[1].imshow(imagem_rgb); axes[1].set_title('Original');           axes[1].axis('off')
axes[2].imshow(clara_rgb);  axes[2].set_title('Clareada    (+80)');  axes[2].axis('off')

plt.tight_layout()
plt.show()