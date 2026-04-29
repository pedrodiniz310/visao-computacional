"""
Visão Computacional — Visualização do Espaço de Cor HSV
Exibe os três canais H, S, V da imagem frutas.png
e demonstra como o Hue representa a cor pura.
"""

import os
import cv2
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/frutas.png'))

# ── Carregamento e conversão ──────────────────────────────────────────────────────
imagem_bgr = cv2.imread(PATH)
if imagem_bgr is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# Conversão BGR → RGB para exibir corretamente com matplotlib
imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

# Conversão BGR → HSV:
# H (Hue/Matiz): 0-179 no OpenCV (360° / 2 para caber em 8 bits)
# S (Saturation/Saturação): 0=cinza puro, 255=cor pura e viva
# V (Value/Brilho): 0=preto absoluto, 255=brilho máximo
# Vantagem: H é INDEPENDENTE de brilho e saturação → ideal para segmentar por cor
imagem_hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

# cv2.split(): separa os 3 canais em arrays individuais de 1 canal cada
h, s, v = cv2.split(imagem_hsv)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Espaço de Cor HSV — Frutas', fontsize=13)

axes[0, 0].imshow(imagem_rgb)
axes[0, 0].set_title('Imagem Original (RGB)'); axes[0, 0].axis('off')

axes[0, 1].imshow(h, cmap='hsv')
axes[0, 1].set_title('H — Matiz (Hue)\n0-179 no OpenCV'); axes[0, 1].axis('off')

axes[1, 0].imshow(s, cmap='gray')
axes[1, 0].set_title('S — Saturação\n0=cinza, 255=cor pura'); axes[1, 0].axis('off')

axes[1, 1].imshow(v, cmap='gray')
axes[1, 1].set_title('V — Valor (Brilho)\n0=preto, 255=máximo'); axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

print('Faixas de Hue no OpenCV (0-179):')
print('  Vermelho : 0-10  e  165-179')
print('  Laranja  : 11-25')
print('  Amarelo  : 26-34')
print('  Verde    : 35-85')
print('  Ciano    : 86-99')
print('  Azul     : 100-130')
print('  Roxo/Rosa: 131-164')