"""
Visão Computacional — Compressão Lossy vs. Lossless
Salva a imagem Lena como JPEG (lossy) e PNG (lossless),
compara tamanhos de arquivo e o impacto nos pixels.
"""

import os
import cv2
import numpy as np

PATH_ORIG = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/lena.png'))
SAIDA_DIR = os.path.dirname(PATH_ORIG)  # salva na pasta images/

PATH_JPEG = os.path.join(SAIDA_DIR, 'lena_lossy.jpg')
PATH_PNG  = os.path.join(SAIDA_DIR, 'lena_lossless.png')

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH_ORIG)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH_ORIG}')

# ── Salvar nos dois formatos ──────────────────────────────────────────────────
# cv2.imwrite(caminho, img, params):
#   JPEG: compressão LOSSY (perde informação) — boa relação tamanho/qualidade
#   IMWRITE_JPEG_QUALITY=[0-100]: 100=melhor qualidade/maior arquivo, 0=pior qualidade
#   50% → arquivo ~5-10x menor que PNG, com artefatos de compressão visíveis
cv2.imwrite(PATH_JPEG, imagem, [cv2.IMWRITE_JPEG_QUALITY, 50])  # lossy: qualidade 50%
# PNG: compressão LOSSLESS (sem perda) — reconstrução pixel-a-pixel idêntica ao original
# Não usa parâmetro de qualidade — apenas nível de compressão ZIP (default)
cv2.imwrite(PATH_PNG,  imagem)                                   # lossless

# ── Comparar tamanhos de arquivo ──────────────────────────────────────────────
tam_orig = os.path.getsize(PATH_ORIG) / 1024
tam_jpeg = os.path.getsize(PATH_JPEG) / 1024
tam_png  = os.path.getsize(PATH_PNG)  / 1024

print('=== Comparação de Tamanhos de Arquivo ===')
print(f'Original  (PNG) : {tam_orig:.1f} KB')
print(f'JPEG q=50 (lossy)   : {tam_jpeg:.1f} KB  →  redução de {100*(1 - tam_jpeg/tam_orig):.1f}%')
print(f'PNG novo  (lossless): {tam_png:.1f}  KB  →  sem perda de informação')

# ── Comparar pixels ───────────────────────────────────────────────────────────
jpeg_recarregado = cv2.imread(PATH_JPEG)
diferenca = cv2.absdiff(imagem, jpeg_recarregado)

print(f'\nDiferença média de pixel (JPEG vs Original): {diferenca.mean():.2f}')
print('→ Valor zero = sem diferença (PNG é idêntico ao original)')
print('→ Valor > 0  = artefatos criados pela compressão JPEG')

# ── Exibição ──────────────────────────────────────────────────────────────────
cv2.imshow('Original PNG (lossless)', imagem)
cv2.imshow('JPEG q=50  (lossy)',      jpeg_recarregado)
cv2.imshow('Diferença amplificada',   cv2.convertScaleAbs(diferenca, alpha=5))
cv2.waitKey(0)
cv2.destroyAllWindows()