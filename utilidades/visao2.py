"""
Visão Computacional — Aula 02
Manipulação de matrizes NumPy: acessar, modificar e operar pixels.
"""

import os
import cv2
import numpy as np

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/lena.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Acesso a pixels ───────────────────────────────────────────────────────────
print('=== Acesso e Modificação de Pixels ===')
# imagem[linha, coluna]: indexado por (y, x) — linha=y, coluna=x
# Retorna array [B, G, R] para imagem colorida
pixel = imagem[100, 100]  # pixel na linha 100, coluna 100
print(f'Pixel (100,100): B={pixel[0]}, G={pixel[1]}, R={pixel[2]}')

# ── Região de interesse (ROI) ───────────────────────────────────────────────────
# imagem[y1:y2, x1:x2] = slice (subimagem) — uma VIEW do array original
# .copy() cria uma cópia independente: modificar roi não altera a imagem original
roi = imagem[100:200, 100:200].copy()
print(f'ROI shape: {roi.shape}')

# ── Operação ponto a ponto: escurecer e clarear ───────────────────────────────
# cv2.subtract e cv2.add: respeitam SATURAÇÃO (0–1255), sem overflow/underflow
# np.full(shape, 80, uint8): cria array do mesmo tamanho da imagem com todos os valores = 80
escura  = cv2.subtract(imagem, np.full(imagem.shape, 80, dtype=np.uint8))
clara   = cv2.add     (imagem, np.full(imagem.shape, 80, dtype=np.uint8))

# ── Exibição ──────────────────────────────────────────────────────────────────
cv2.imshow('Original', imagem)
cv2.imshow('Escurecida (−80)', escura)
cv2.imshow('Clareada  (+80)', clara)
cv2.waitKey(0)
cv2.destroyAllWindows()