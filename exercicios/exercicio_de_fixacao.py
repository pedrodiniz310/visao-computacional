"""
A1/2: Segmentação e Extração de Metadados

Script de fixação para praticar o pipeline completo de visão computacional:
1. Carregar imagem
2. Converter para cinza
3. Filtrar ruído
4. Binarizar com Otsu
5. Aplicar morfologia
6. Encontrar contornos com hierarquia
7. Calcular métricas por objeto (área, perímetro, centroide, Momentos de Hu, furos)
"""

import os
import cv2
import numpy as np

# ── Carregamento ──────────────────────────────────────────────────────────────
# Caminho relativo à pasta images/ (um nível acima de exercicios/)
imagem = cv2.imread(os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/engrenagem.png')))
if imagem is None:
    print('Erro: imagem não encontrada!')
    exit(1)

# ── Conversão para escala de cinza ────────────────────────────────────────────
# Necessário para filtros e binarização (operações em 1 canal)
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── ETAPA 1: SEGMENTAÇÃO ──────────────────────────────────────────────────────

# Filtrar ruído
# Use medianBlur para ruído sal-e-pimenta (mais comum nas provas)
# Use GaussianBlur para ruído gaussiano/eletrônico
# O kernel ímpar (5) define o tamanho da vizinhança analisada
filtrada = cv2.medianBlur(cinza, 5)          # k ímpar: 3, 5, 7, 9
# filtrada = cv2.GaussianBlur(cinza, (5,5), 0)  # alternativa para ruído gaussiano

# Binarizar com Otsu (escolhe o limiar automaticamente analisando o histograma)
# THRESH_BINARY: pixel > limiar → 255 (branco); pixel ≤ limiar → 0 (preto)
# O "_" descarta o limiar retornado (não necessário aqui)
_, binaria = cv2.threshold(filtrada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Se a imagem tiver fundo claro e objeto escuro, use THRESH_BINARY_INV:
# _, binaria = cv2.threshold(filtrada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ── MORFOLOGIA — elemento estruturante ───────────────────────────────────────
# MORPH_ELLIPSE: melhor para formas circulares (como engrenagens)
# (5, 5): tamanho do kernel — kernels maiores têm efeito mais amplo
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Abertura (MORPH_OPEN = erosão → dilatação):
# Remove pontos brancos isolados no fundo (ruído pós-binarização)
# sem alterar significativamente os objetos principais
limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

# Fechamento (MORPH_CLOSE = dilatação → erosão):
# Preenche buracos e imperfeições pequenas dentro do objeto
# (conecta regiões próximas do mesmo objeto)
limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, kernel)

# ── ETAPA 2: EXTRAÇÃO DE CARACTERÍSTICAS ─────────────────────────────────────

# Encontrar contornos com hierarquia de 2 níveis (RETR_CCOMP):
# Nível 0: objetos externos
# Nível 1: furos dentro dos objetos (cavidades)
# hierarquia[0][i] = [próximo_irmão, irmão_anterior, primeiro_filho, pai]
# Se hierarquia[0][i][3] == -1 → sem pai → contorno externo (objeto)
# Se hierarquia[0][i][3] != -1 → tem pai → furo dentro de um objeto
contornos, hierarquia = cv2.findContours(
    limpa,
    cv2.RETR_CCOMP,          # 2 níveis: objetos (nível 0) + furos (nível 1)
    cv2.CHAIN_APPROX_SIMPLE  # comprime pontos redundantes (economiza memória)
)

if len(contornos) == 0:
    print('Nenhum contorno encontrado!')
    exit(1)

print(f'Objetos encontrados: {len(contornos)}')
print('=' * 50)

# ── Processar cada contorno externo (objetos, não furos) ──────────────────────
for i, contorno in enumerate(contornos):
    # Ignorar furos (contornos que têm pai != -1 são furos)
    if hierarquia[0][i][3] != -1:
        continue

    print(f'\n--- Objeto {i} ---')

    # ── Métricas DIMENSIONAIS ──────────────────────────────────────────────
    # cv2.contourArea(): número de pixels dentro do contorno (em px²)
    area      = cv2.contourArea(contorno)
    # cv2.arcLength(contorno, fechado): comprimento do contorno em pixels
    perimetro = cv2.arcLength(contorno, True)  # True = contorno fechado
    print(f'Área:      {int(area)} px²')
    print(f'Perímetro: {int(perimetro)} px')

    # ── Métricas INERCIAIS (centroide) ────────────────────────────────────
    # cv2.moments(): calcula os momentos geométricos do contorno
    # m00 = área; m10 = soma de x; m01 = soma de y
    # Centroide: cx = m10/m00 (x médio); cy = m01/m00 (y médio)
    M = cv2.moments(contorno)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print(f'Centroide: ({cx}, {cy})')
    else:
        cx, cy = 0, 0
        print('Centroide: indisponível (área zero)')

    # ── Momentos de Hu ────────────────────────────────────────────────────
    # cv2.HuMoments(): 7 valores invariantes à rotação, escala e translação
    # Usados para comparar formas independente de transformações geométricas
    hu = cv2.HuMoments(M).flatten()
    print('Momentos de Hu:')
    for j, h in enumerate(hu):
        print(f'  Hu[{j}] = {h:.6e}')

    # ── Métricas TOPOLÓGICAS — contar furos deste objeto ──────────────────
    # Varre todos os contornos e conta os que têm este objeto como pai
    furos = 0
    for j in range(len(contornos)):
        if hierarquia[0][j][3] == i:   # o pai do contorno j é o objeto i
            furos += 1
    print(f'Furos/cavidades: {furos}')

    # ── Anotar na imagem ──────────────────────────────────────────────────
    cv2.drawContours(imagem, [contorno], -1, (0, 255, 0), 2)       # contorno verde
    cv2.circle(imagem, (cx, cy), 5, (0, 0, 255), -1)               # centroide vermelho
    cv2.putText(imagem, f'A={int(area)}', (cx + 8, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)   # área em amarelo

print('=' * 50)

# ── Exibição final ────────────────────────────────────────────────────────────
cv2.imshow('Segmentação e Métricas — Engrenagem', imagem)
cv2.waitKey(0)       # aguarda qualquer tecla
cv2.destroyAllWindows()
