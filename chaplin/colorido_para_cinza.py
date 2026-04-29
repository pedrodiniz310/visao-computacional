"""
Visão Computacional — Conversão Colorido para Cinza
Converte a imagem do Chaplin de BGR para escala de cinza
e exibe as duas versões lado a lado.
"""

import os
import cv2
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/chaplin.jpg'))

# ── Carregamento ──────────────────────────────────────────────────────────────
# cv2.imread() lê a imagem como array NumPy no formato BGR (Blue-Green-Red)
# A ordem BGR é padrão do OpenCV (diferente do RGB do matplotlib/pillow)
imagem_bgr = cv2.imread(PATH)
if imagem_bgr is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# cv2.COLOR_BGR2RGB: inverte a ordem dos canais BGR → RGB
# Necessário porque matplotlib.imshow() espera RGB, não BGR
imagem_rgb  = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

# cv2.COLOR_BGR2GRAY: converte 3 canais (B,G,R) para 1 canal usando a fórmula de luminância:
# Y = 0.114*B + 0.587*G + 0.299*R
# O canal verde tem maior peso pois o olho humano é mais sensível ao verde
imagem_gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Chaplin — Colorido vs. Escala de Cinza', fontsize=13)

ax1.imshow(imagem_rgb);              ax1.set_title('Original (RGB)');      ax1.axis('off')
ax2.imshow(imagem_gray, cmap='gray'); ax2.set_title('Escala de Cinza');     ax2.axis('off')

plt.tight_layout()
plt.show()

print(f'Original : shape={imagem_bgr.shape}, dtype={imagem_bgr.dtype}')
print(f'Cinza    : shape={imagem_gray.shape}, dtype={imagem_gray.dtype}')
print('→ Converter para cinza reduz de 3 canais para 1 canal, economizando memória.')