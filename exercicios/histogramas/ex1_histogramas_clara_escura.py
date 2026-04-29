"""
Exercício 1 — Histogramas: Imagem Clara vs. Escura
Carregue uma imagem clara e uma escura. Plote os histogramas de ambas e
identifique em qual região do gráfico (esquerda ou direita) a concentração
de pixels é maior.
"""

import os
import cv2
import matplotlib.pyplot as plt

# ── Caminhos das imagens ──────────────────────────────────────────────────────
PATH_CLARA  = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/frutas_clara.png'))
PATH_ESCURA = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/frutas_escura.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
clara  = cv2.imread(PATH_CLARA,  cv2.IMREAD_GRAYSCALE)
escura = cv2.imread(PATH_ESCURA, cv2.IMREAD_GRAYSCALE)

if clara is None or escura is None:
    raise FileNotFoundError('Imagem(ns) não encontrada(s). Verifique os caminhos.')

# ── Cálculo dos histogramas ───────────────────────────────────────────────────
# cv2.calcHist([imgs], [canais], máscara, [nBins], [range]):
#   [0] = canal único (imagem em escala de cinza)
#   None = sem máscara (processa todos os pixels)
#   [256] = 256 bins (um por nível de intensidade 0-255)
#   [0, 256] = inclui o valor 0, exclui o 256
# Retorna: array (256, 1) com contagem de pixels por intensidade
hist_clara  = cv2.calcHist([clara],  [0], None, [256], [0, 256])
hist_escura = cv2.calcHist([escura], [0], None, [256], [0, 256])

# ── Plotagem ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Histogramas — Imagem Clara vs. Escura', fontsize=14)

axes[0, 0].imshow(clara, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Imagem Clara')
axes[0, 0].axis('off')

axes[0, 1].imshow(escura, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Imagem Escura')
axes[0, 1].axis('off')

axes[1, 0].plot(hist_clara, color='steelblue')
axes[1, 0].set_title('Histograma — Clara\n(concentração à DIREITA: tons altos / claros)')
axes[1, 0].set_xlabel('Nível de intensidade (0-255)')
axes[1, 0].set_ylabel('Número de pixels')
axes[1, 0].set_xlim([0, 256])

axes[1, 1].plot(hist_escura, color='darkred')
axes[1, 1].set_title('Histograma — Escura\n(concentração à ESQUERDA: tons baixos / escuros)')
axes[1, 1].set_xlabel('Nível de intensidade (0-255)')
axes[1, 1].set_ylabel('Número de pixels')
axes[1, 1].set_xlim([0, 256])

plt.tight_layout()
plt.show()

# ── Análise textual ───────────────────────────────────────────────────────────
print('=== Análise dos Histogramas ===')
print()
print('Imagem CLARA:')
print('  → A maior concentração de pixels está à DIREITA do histograma')
print('  → Isso indica que a maioria dos pixels possui intensidades ALTAS (próximas de 255)')
print('  → Visualmente, os tons predominantes são brancos e cinzas claros.')
print()
print('Imagem ESCURA:')
print('  → A maior concentração de pixels está à ESQUERDA do histograma')
print('  → Isso indica que a maioria dos pixels possui intensidades BAIXAS (próximas de 0)')
print('  → Visualmente, os tons predominantes são pretos e cinzas escuros.')
