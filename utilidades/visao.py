"""
Visão Computacional — Aula 01
Leitura, exibição e propriedades de uma imagem digital.
"""

import os
import cv2

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/lena.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
# cv2.imread(caminho) retorna um array NumPy de shape (altura, largura, 3)
# Os canais estão na ordem BGR (Blue-Green-Red), não RGB!
# Retorna None se o arquivo não existir ou o caminho estiver errado.
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Propriedades da imagem ──────────────────────────────────────────────────────
# imagem.shape retorna (altura, largura, n_canais) — formato (linhas, colunas, canais)
# imagem.dtype = tipo de dado: uint8 = inteiro sem sinal de 8 bits (valores 0–1255)
# imagem.min() / max(): valor mínimo e máximo de TODOS os pixels de TODOS os canais
print(f'Tipo Python  : {type(imagem)}')
print(f'Dimensões    : {imagem.shape}  →  (altura, largura, canais)')
print(f'Tipo de dado : {imagem.dtype}')
print(f'Nº de pixels : {imagem.shape[0] * imagem.shape[1]}')
print(f'Valor mín.   : {imagem.min()}')
print(f'Valor máx.   : {imagem.max()}')

# ── Exibição ──────────────────────────────────────────────────────────
# cv2.imshow(nome_janela, imagem): exibe a imagem em uma janela do sistema
# cv2.waitKey(0): aguarda indefinidamente até alguma tecla ser pressionada
# cv2.destroyAllWindows(): fecha todas as janelas abertas pelo OpenCV
cv2.imshow('Lena — Imagem Original', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()