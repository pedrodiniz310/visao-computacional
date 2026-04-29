"""
Exercício 3 — Restauração da Impressão Digital
Utilize métodos de pré-processamento e segmentação de imagens para restaurar
a imagem fingerprint.jpg.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/fingerprint.jpg'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Pipeline de restauração ───────────────────────────────────────────────────

# Passo 1: Equalização CLAHE — melhora o contraste LOCAL das cristas
# clipLimit=3.0: permite até 3× a altura média do histograma (mais contraste que o padrão 2.0)
# tileGridSize=(8,8): divide em 64 tiles para equalização independente
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
equalizada = clahe.apply(cinza)

# Passo 2: Filtro Gaussiano para suavizar ruído de fundo sem destruir cristas
# Kernel 3×3 pequeno: preserva detalhes finos das cristas da digital
suave = cv2.GaussianBlur(equalizada, (3, 3), 0)

# Passo 3: Realce das cristas por subtração de fundo (filtro passa-alta)
# Fundo estimado com blur muito forte (31×31) → captura apenas variações lentas
fundo = cv2.GaussianBlur(suave, (31, 31), 0)
# Subtrai o fundo: realça variações locais (cristas e vales)
# +128: centraliza o resultado em 128 (evita clípping em 0)
realcada = cv2.subtract(suave, fundo) + 128  # centraliza em 128

# Passo 4: Limiarização adaptativa — binariza as cristas
# THRESH_BINARY_INV: cristas escuras → branco (facilita morfologia)
# blockSize=13: vizinhança 13×13 para calcular limiar local
# C=4: constante subtraída para afinar a detecção
bin_adapt = cv2.adaptiveThreshold(suave, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 13, 4)

# Passo 5: Morfologia de refinamento
# MORPH_CROSS: cruz 3×3 — preserva direção das cristas (melhor que retangular)
kernel_thin = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# Fechamento: conecta cristas fragmentadas
fechamento = cv2.morphologyEx(bin_adapt, cv2.MORPH_CLOSE, kernel_thin, iterations=1)
# Abertura: remove pontos isolados
abertura   = cv2.morphologyEx(fechamento, cv2.MORPH_OPEN, kernel_thin, iterations=1)

# Passo 6: Erosão leve para afinar as cristas (esqueletização simples)
eroded = cv2.erode(abertura, kernel_thin, iterations=1)

# ── Exibição do pipeline completo ────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Pipeline de Restauração de Impressão Digital', fontsize=13)

etapas = [
    (cinza,      'Original (cinza)'),
    (equalizada, 'CLAHE (contraste local)'),
    (suave,      'Gaussiano 3x3\n(suavização)'),
    (realcada,   'Realce de cristas\n(high-pass)'),
    (bin_adapt,  'Binarização Adaptativa\n(cristas isoladas)'),
    (fechamento, 'Fechamento\n(conecta cristas)'),
    (abertura,   'Abertura\n(remove ruído)'),
    (eroded,     'Erosão\n(afina cristas)'),
]

for ax, (img, titulo) in zip(axes.flat, etapas):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(titulo, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()

# ── Comparação final ──────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle('Resultado da Restauração — Comparação', fontsize=13)

axes2[0].imshow(cinza,    cmap='gray'); axes2[0].set_title('Original'); axes2[0].axis('off')
axes2[1].imshow(bin_adapt, cmap='gray'); axes2[1].set_title('Binarização Adaptativa'); axes2[1].axis('off')
axes2[2].imshow(eroded,   cmap='gray'); axes2[2].set_title('Resultado Final (pipeline completo)'); axes2[2].axis('off')

plt.tight_layout()
plt.show()

print('=== Pipeline de Restauração da Impressão Digital ===')
print()
print('Passo 1 — CLAHE:')
print('  → Equaliza o contraste LOCALMENTE em blocos 8x8.')
print('  → Cristas pouco contrastadas ficam mais visíveis sem saturar as boas áreas.')
print()
print('Passo 2 — Filtro Gaussiano (leve):')
print('  → Remove o ruído de alta frequência (grãos, sujeira) sem borrar as cristas.')
print('  → Kernel pequeno (3x3) para não perder detalhes de impressão digital.')
print()
print('Passo 3 — Realce de cristas (High-Pass):')
print('  → Subtrai uma versão com muito blur (fundo estimado) da imagem suave.')
print('  → Equivale a um filtro passa-alta: elimina gradientes de iluminação global')
print('    e preserva apenas as variações locais das cristas.')
print()
print('Passo 4 — Limiarização Adaptativa:')
print('  → Binariza as cristas usando o contexto local (blocos 13x13).')
print('  → Robusta contra iluminação não-uniforme (comum em digitais envelhecidas).')
print('  → THRESH_BINARY_INV: cristas ficam brancas (255), fundo preto (0).')
print()
print('Passo 5 — Morfologia:')
print('  → Fechamento: conecta cristas com pequenas interrupções.')
print('  → Abertura:   elimina pontos isolados que são ruído.')
print('  → Erosão:     afina as cristas para aproximar da largura original.')
