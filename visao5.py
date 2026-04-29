"""
Visão Computacional — Hierarquia de Contornos e Número de Euler
Carrega a imagem furo1.png, detecta todos os contornos com hierarquia
completa (RETR_TREE) e conta objetos vs furos para calcular o
número de Euler: E = Objetos − Furos.
"""

import os
import cv2
import numpy as np
import math  # disponível para cálculos matemáticos adicionais se necessário

# Constrói o caminho absoluto para a imagem a partir do diretório deste script
PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), './images/furo1.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
# cv2.imread() carrega a imagem como array NumPy em formato BGR (3 canais)
imagem = cv2.imread(PATH)
if imagem is None:
    print('Erro: Imagem não encontrada. Verifique o caminho e o nome do arquivo.')
    exit(1)

# ── Conversão para escala de cinza ────────────────────────────────────────────
# cv2.findContours() requer imagem em 1 canal (escala de cinza ou binária)
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Binarização ───────────────────────────────────────────────────────────────
# cv2.threshold() divide os pixels em 2 grupos pelo limiar 127:
# pixels > 127 → 255 (branco = objeto); pixels ≤ 127 → 0 (preto = fundo)
# "ret" recebe o limiar usado; "imgBinarizada" é o resultado binário
ret, imgBinarizada = cv2.threshold(imagemCinza, 127, 255, cv2.THRESH_BINARY)

# ── Contornos com hierarquia completa ─────────────────────────────────────────
# RETR_TREE: recupera TODOS os contornos e reconstrói a hierarquia completa em árvore.
#   - Nível 0: contornos externos (objetos)
#   - Nível 1: contornos dentro de objetos (furos)
#   - Nível 2+: contornos dentro de furos (ilhas dentro de furos), etc.
# Diferente de RETR_CCOMP (2 níveis) ou RETR_EXTERNAL (só externo)
#
# hierarquia[0][i] = [próximo, anterior, filho, pai]
#   hierarquia[0][i][3] == -1  → sem pai → contorno EXTERNO (objeto)
#   hierarquia[0][i][3] != -1  → tem pai → contorno INTERNO (furo ou ilha)
contornos, hierarquia = cv2.findContours(imgBinarizada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Exibe a estrutura completa de hierarquia para análise
print(hierarquia)

# ── Contagem de objetos e furos ───────────────────────────────────────────────
objetos = 0  # contornos sem pai (externos) → objetos sólidos
furos   = 0  # contornos com pai (internos) → buracos dentro de objetos

for i in range(len(contornos)):
    if hierarquia[0][i][3] == -1:  # campo "pai" == -1 → contorno externo (objeto)
        objetos += 1
    else:                           # campo "pai" != -1 → contorno interno (furo)
        furos += 1

# ── Número de Euler ───────────────────────────────────────────────────────────
# O Número de Euler (E) é uma característica topológica do objeto:
#   E = Objetos − Furos
# Interpretação:
#   E = 1  → objeto sólido sem furos (ex: círculo cheio)
#   E = 0  → um objeto com um furo (ex: anel/donuts)
#   E = -1 → um objeto com dois furos
# Usado em inspeção industrial para contar furos em peças.
euler = objetos - furos + 1

# ── Resultados ────────────────────────────────────────────────────────────────
print(f'Objetos:        {objetos}')  # contornos externos detectados
print(f'Furos:          {furos}')    # contornos internos (buracos)
print(f'Número de Euler: {euler}')   # métrica topológica
