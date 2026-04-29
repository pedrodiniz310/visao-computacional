import cv2
import numpy as np

# ============================================================
# A1/2: Segmentação e Extração de Metadados
# ============================================================

# carregar imagem
imagem = cv2.imread('../imagens/engrenagem.png')
if imagem is None:
    print('Erro: imagem não encontrada!')
    exit(1)

# converter para cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# etapa 1: segmentação

# filtrar ruído
# Use medianBlur para ruído sal-e-pimenta (mais comum nas provas)
# Use GaussianBlur para ruído gaussiano/eletrônico
filtrada = cv2.medianBlur(cinza, 5)          # k ímpar: 3, 5, 7, 9
# filtrada = cv2.GaussianBlur(cinza, (5,5), 0)

# binarizar imagem (Otsu escolhe o limiar automaticamente)
_, binaria = cv2.threshold(filtrada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Se a imagem tiver fundo claro e objeto escuro, inverta:
# _, binaria = cv2.threshold(filtrada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 5. MORFOLOGIA — elemento estruturante
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Abertura: remove pontos brancos isolados no fundo (ruído pós-binarização)
limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

# Fechamento: preenche buracos e imperfeições dentro do objeto
limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, kernel)

# --- ETAPA 2: EXTRAÇÃO DE CARACTERÍSTICAS ---

# 6. ENCONTRAR CONTORNOS COM HIERARQUIA (necessário para detectar furos)
contornos, hierarquia = cv2.findContours(
    limpa,
    cv2.RETR_CCOMP,          # 2 níveis: objetos (nível 0) + furos (nível 1)
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contornos) == 0:
    print('Nenhum contorno encontrado!')
    exit(1)

print(f'Objetos encontrados: {len(contornos)}')
print('=' * 50)

# 7. PROCESSAR CADA CONTORNO EXTERNO (objetos, não furos)
for i, contorno in enumerate(contornos):
    # Ignorar furos (contornos que têm pai != -1 são furos)
    if hierarquia[0][i][3] != -1:
        continue

    print(f'\n--- Objeto {i} ---')

    # --- DIMENSIONAIS ---
    area      = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    print(f'Área:      {int(area)} px²')
    print(f'Perímetro: {int(perimetro)} px')

    # --- INERCIAIS ---
    M = cv2.moments(contorno)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print(f'Centroide: ({cx}, {cy})')
    else:
        cx, cy = 0, 0
        print('Centroide: indisponível (área zero)')

    # Momentos de Hu (7 valores invariantes à rotação, escala e translação)
    hu = cv2.HuMoments(M).flatten()
    print('Momentos de Hu:')
    for j, h in enumerate(hu):
        print(f'  Hu[{j}] = {h:.6e}')

    # --- TOPOLÓGICAS — contar furos deste objeto ---
    furos = 0
    for j in range(len(contornos)):
        if hierarquia[0][j][3] == i:   # pai do contorno j é o objeto i
            furos += 1
    print(f'Furos/cavidades: {furos}')

    # Desenhar na imagem
    cv2.drawContours(imagem, [contorno], -1, (0, 255, 0), 2)
    cv2.circle(imagem, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(imagem, f'A={int(area)}', (cx + 8, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

print('=' * 50)

# --- VISUALIZAÇÃO ---
cv2.imshow('Original', imagem)
cv2.imshow('Cinza', cinza)
cv2.imshow('Filtrada', filtrada)
cv2.imshow('Binarizada', binaria)
cv2.imshow('Limpa (mascara final)', limpa)
cv2.waitKey(0)
cv2.destroyAllWindows()
