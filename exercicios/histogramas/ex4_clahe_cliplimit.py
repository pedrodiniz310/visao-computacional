"""
Exercício 4 — CLAHE com diferentes valores de clipLimit

Encontre uma imagem com iluminação mista (áreas muito claras e muito escuras).
Aplique o CLAHE com diferentes valores de clipLimit (1.0, 2.0 e 5.0) e descreva
o impacto visual no ruído da imagem.
"""

import os
import cv2
import matplotlib.pyplot as plt

# Caminho absoluto da imagem com iluminação mista (alta variação de contraste)
PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/estrada_escura.png'))

# ── Carregamento e conversão para cinza ───────────────────────────────────────
# CLAHE opera em 1 canal (escala de cinza); o canal V no espaço HSV para colorido
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# cv2.COLOR_BGR2GRAY: combina os 3 canais usando a fórmula de luminância:
# Y = 0.299*R + 0.587*G + 0.114*B
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Equalização global (referência) ───────────────────────────────────────────
# cv2.equalizeHist(): redistribui os pixels para cobrir uniformemente [0-255]
# Problema: trata a imagem inteira como uma só → over-amplifica regiões já claras
# e pode criar artefatos em imagens com iluminação não-uniforme
equalizada_global = cv2.equalizeHist(cinza)

# ── CLAHE com diferentes clipLimits ───────────────────────────────────────────
# CLAHE = Contrast Limited Adaptive Histogram Equalization
# Diferença do equalizeHist global:
#   1. Divide a imagem em tiles (regiões) de tamanho tileGridSize
#   2. Equaliza cada tile INDEPENDENTEMENTE → adapta ao contraste LOCAL
#   3. clipLimit: limita o histograma para EVITAR amplificação excessiva de ruído
#      - clipLimit baixo (1.0): pouco contraste → mais suave, menos ruído
#      - clipLimit alto (5.0): muito contraste → bordas mais nítidas, mais ruído
# tileGridSize=(8,8): divide a imagem em 8×8 = 64 regiões independentes
resultados = {}
clip_limits = [1.0, 2.0, 5.0]  # três níveis de limitação de contraste

for clip in clip_limits:
    # Cria objeto CLAHE com o clipLimit especificado
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    # Aplica o CLAHE na imagem em escala de cinza
    resultados[clip] = clahe.apply(cinza)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('CLAHE — Impacto do clipLimit na imagem com iluminação mista', fontsize=13)

# Linha superior: imagens resultantes
axes[0, 0].imshow(cinza,             cmap='gray'); axes[0, 0].set_title('Original (cinza)');            axes[0, 0].axis('off')
axes[0, 1].imshow(equalizada_global, cmap='gray'); axes[0, 1].set_title('Equalização Global');          axes[0, 1].axis('off')
axes[0, 2].imshow(resultados[1.0],   cmap='gray'); axes[0, 2].set_title('CLAHE clipLimit=1.0\n(suave, menos ruído)'); axes[0, 2].axis('off')

# Linha inferior: CLAHE 2.0 e 5.0 + histograma comparativo
axes[1, 0].imshow(resultados[2.0], cmap='gray'); axes[1, 0].set_title('CLAHE clipLimit=2.0\n(equilíbrio ruído/contraste)'); axes[1, 0].axis('off')
axes[1, 1].imshow(resultados[5.0], cmap='gray'); axes[1, 1].set_title('CLAHE clipLimit=5.0\n(mais contraste, mais ruído)'); axes[1, 1].axis('off')

# Histograma comparativo: mostra a distribuição de intensidades para cada método
axes[1, 2].plot(cv2.calcHist([cinza],           [0], None, [256], [0, 256]), label='Original', color='black')
axes[1, 2].plot(cv2.calcHist([resultados[1.0]], [0], None, [256], [0, 256]), label='clip=1.0',  color='blue')
axes[1, 2].plot(cv2.calcHist([resultados[2.0]], [0], None, [256], [0, 256]), label='clip=2.0',  color='green')
axes[1, 2].plot(cv2.calcHist([resultados[5.0]], [0], None, [256], [0, 256]), label='clip=5.0',  color='red')
axes[1, 2].set_title('Histogramas comparativos')
axes[1, 2].set_xlim([0, 256])
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# ── Análise ───────────────────────────────────────────────────────────────────
print('=== Impacto do clipLimit no CLAHE ===')
print()
print('O CLAHE (Contrast Limited Adaptive Histogram Equalization) divide a imagem')
print('em tiles e equaliza cada um independentemente, limitando a amplificação')
print('de contraste pelo parâmetro clipLimit.')
print()
print('clipLimit=1.0: mínima amplificação → resultado mais suave, menos ruído,')
print('              mas contraste local pouco melhorado.')
print()
print('clipLimit=2.0: equilíbrio → boa melhora de contraste sem amplificar ruído.')
print('              Valor padrão recomendado para a maioria dos casos.')
print()
print('clipLimit=5.0: alta amplificação → muito contraste e detalhes visíveis,')
print('              mas ruído também amplificado → mais granulado.')
