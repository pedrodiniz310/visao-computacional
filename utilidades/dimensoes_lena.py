"""
Visão Computacional — Dimensões da Imagem Lena
Exibe as dimensões, canais e informações da imagem padrão de teste Lena.
"""

import os
import cv2

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/lena.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

alt, larg, canais = imagem.shape

# ── Informações ───────────────────────────────────────────────────────────────
# imagem.shape = (altura, largura, canais) — em pixels
# imagem.nbytes = total de bytes na memória = alt * larg * canais * bytes_por_pixel
#   uint8 = 1 byte/pixel/canal → 512*512*3 = 786 432 bytes ≈ 768 KB
print('=== Dimensões da Imagem Lena ===')
print(f'Largura  : {larg} pixels')
print(f'Altura   : {alt}  pixels')
print(f'Canais   : {canais} (BGR)')
print(f'Total    : {alt * larg} pixels')
print(f'Dtype    : {imagem.dtype}')
print(f'Tamanho em memória (aprox.): {imagem.nbytes / 1024:.1f} KB')

# ── Exibição ──────────────────────────────────────────────────────────────────
cv2.imshow(f'Lena  {larg}x{alt}', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()