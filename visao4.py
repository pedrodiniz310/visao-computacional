"""
Visão Computacional — Análise de Forma com Contornos
Carrega a imagem quadrado.png, binariza, extrai o contorno
e calcula métricas geométricas: área, perímetro, proporção,
extensão, solidez, caixa delimitadora e envoltória convexa (hull).
"""

import os
import cv2
import numpy as np

# Constrói o caminho absoluto para a imagem a partir do diretório deste script
PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), './images/quadrado.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
# cv2.imread() retorna um array NumPy (BGR). Retorna None se o arquivo não existir.
imagem = cv2.imread(PATH)
if imagem is None:
    print('Erro: Imagem não encontrada. Verifique o caminho e o nome do arquivo.')
    exit(1)

# ── Conversão para escala de cinza ────────────────────────────────────────────
# cv2.findContours() opera em imagens de 1 canal; precisamos de escala de cinza.
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Binarização ───────────────────────────────────────────────────────────────
# cv2.threshold(src, limiar, maxVal, tipo) → (retVal, imagemBinarizada)
# Limiar 127 = meio da escala [0-255]; pixels > 127 → 255 (branco), ≤ 127 → 0
# THRESH_BINARY: retorna imagem binária (0 ou 255)
# O retorno "_" descarta o valor do limiar (relevante apenas no método Otsu)
_, imgBinarizada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY)

# ── Encontrar contornos ───────────────────────────────────────────────────────
# cv2.findContours() detecta os contornos dos objetos binários.
# RETR_EXTERNAL: retorna apenas os contornos mais externos (ignora furos internos)
# CHAIN_APPROX_SIMPLE: comprime segmentos horizontais/verticais/diagonais (economiza memória)
# Retorna: lista de contornos e hierarquia (aqui descartada com "_")
contornos, _ = cv2.findContours(imgBinarizada,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

if len(contornos) == 0:
    print('Nenhum contorno encontrado na imagem.')
    exit(1)

# Pega o primeiro (e principal) contorno da imagem
contorno = contornos[0]

# ── Caixa delimitadora (Bounding Rectangle) ───────────────────────────────────
# cv2.boundingRect() retorna o retângulo que envolve o contorno (x, y, largura, altura)
# (x, y) é o canto superior esquerdo; (x+w, y+h) é o canto inferior direito
x, y, w, h = cv2.boundingRect(contorno)
# Desenha o retângulo delimitador na imagem em azul (BGR: 255,0,0), espessura 2
cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)

# ── Envoltória convexa (Convex Hull) ─────────────────────────────────────────
# cv2.convexHull() calcula o menor polígono convexo que envolve todos os pontos do contorno
# Útil para medir a "compacidade" do objeto e comparar com o contorno real
hull = cv2.convexHull(contorno)
# Desenha a envoltória convexa em azul também (poderia usar outra cor para diferençar)
cv2.drawContours(imagem, [hull], -1, (255, 0, 0), 2)

# ── Cálculo das métricas geométricas ──────────────────────────────────────────
# ÁREA DO CONTORNO: número de pixels dentro do contorno (unidade: px²)
area = int(cv2.contourArea(contorno))

# PERÍMETRO: comprimento total do contorno (True = contorno fechado)
perimetro = int(cv2.arcLength(contorno, True))

# ÁREA DA CAIXA DELIMITADORA: w × h (sempre ≥ área do objeto)
area_caixa_delimitadora = w * h

# ÁREA DA ENVOLTÓRIA CONVEXA: sempre ≥ área do objeto
area_envoltoria_convexa = int(cv2.contourArea(hull))

# PROPORÇÃO (Aspect Ratio): largura / altura
# → 1.0 = quadrado perfeito; > 1 = mais largo que alto; < 1 = mais alto que largo
proporcao = w / h

# EXTENSÃO (Extent): área do objeto / área da caixa delimitadora
# → Mede o quão bem o objeto preenche seu bounding box (máx = 1.0 para retângulo perfeito)
extensao = area / area_caixa_delimitadora

# SOLIDEZ (Solidity): área do objeto / área da envoltória convexa
# → Mede o quão convexo é o objeto (1.0 = completamente convexo, < 1 = tem reentrâncias)
solidez = area / area_envoltoria_convexa

# ── Impressão dos resultados ──────────────────────────────────────────────────
print(f'Área do objeto:              {area} px²')
print(f'Perímetro:                   {perimetro} px')
print(f'Proporção (w/h):             {proporcao:.4f}')
print(f'Extensão (area/bbox):        {extensao:.4f}')
print(f'Solidez (area/hull):         {solidez:.4f}')
print(f'Área da caixa delimitadora:  {area_caixa_delimitadora} px²')
print(f'Área da envoltória convexa:  {area_envoltoria_convexa} px²')

# ── Exibição ──────────────────────────────────────────────────────────────────
# cv2.imshow() abre uma janela com o nome indicado; cv2.waitKey(0) aguarda tecla
cv2.imshow('Caixa (azul) e Hull (azul)', imagem)
cv2.waitKey(0)           # 0 = espera indefinidamente por qualquer tecla
cv2.destroyAllWindows()  # fecha todas as janelas do OpenCV
