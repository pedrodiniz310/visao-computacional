"""
Exercício 1 — Filtros Lineares no Ruído Sal e Pimenta

Carregue a imagem sal_e_pimenta.png e aplique os filtros de Média e Gaussiano
com o mesmo tamanho de kernel (7x7). Explique por que esses filtros lineares
apenas "espalham" o ruído em vez de removê-lo completamente.
"""

import os
import cv2
import matplotlib.pyplot as plt

# Caminho absoluto para a imagem (dois níveis acima → pasta images/)
PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/sal_e_pimenta.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
# cv2.IMREAD_GRAYSCALE (=0): carrega direto em 1 canal (8 bits, 0-255)
# O ruído sal-e-pimenta fica mais evidente em escala de cinza
imagem = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

K = 7  # tamanho do kernel — deve ser ÍMPAR para ter um pixel central definido

# ── Aplicar filtros lineares ──────────────────────────────────────────────────
# FILTRO DE MÉDIA (Box Blur):
# cv2.blur(src, ksize): substitui cada pixel pela MÉDIA ARITMÉTICA de todos
# os K×K vizinhos. Todos os vizinhos têm peso IGUAL → peso = 1/(K²).
# Problema: um pixel de ruído (255) "contamina" todos os K² vizinhos,
# criando uma mancha cinza no lugar do pico de ruído → o ruído é ESPALHADO.
media = cv2.blur(imagem, (K, K))

# FILTRO GAUSSIANO:
# cv2.GaussianBlur(src, ksize, sigmaX): usa pesos maiores no centro
# (distribuição gaussiana 2D) e menores nas bordas.
# O parâmetro sigmaX=0 faz o OpenCV calcular sigma automaticamente a partir
# do tamanho do kernel: sigma ≈ 0.3 * ((K-1) * 0.5 - 1) + 0.8
# Resultado: melhor para suavizar textura fina, mas ainda não elimina ruído extremo
gaussiano = cv2.GaussianBlur(imagem, (K, K), 0)

# ── Exibição ──────────────────────────────────────────────────────────────────
# vmin=0, vmax=255: fixa a escala de cor para comparação justa entre imagens
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Filtros Lineares no Ruído Sal-e-Pimenta (kernel {K}x{K})', fontsize=13)

axes[0].imshow(imagem,    cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original — Ruído Sal e Pimenta'); axes[0].axis('off')

axes[1].imshow(media,     cmap='gray', vmin=0, vmax=255)
axes[1].set_title(f'Filtro de Média ({K}x{K})\nEspalha o ruído'); axes[1].axis('off')

axes[2].imshow(gaussiano, cmap='gray', vmin=0, vmax=255)
axes[2].set_title(f'Filtro Gaussiano ({K}x{K})\nReduz, mas não elimina'); axes[2].axis('off')

plt.tight_layout()
plt.show()

# ── Explicação escrita ────────────────────────────────────────────────────────
print('=== Por que filtros lineares apenas "espalham" o ruído Sal e Pimenta? ===')
print()
print('O ruído Sal e Pimenta é caracterizado por pixels com valores EXTREMOS:')
print('  → Sal:     pixels com valor 255 (branco puro)')
print('  → Pimenta: pixels com valor 0  (preto puro)')
print()
print('Filtro de Média:')
print('  → Substitui cada pixel pela MÉDIA ARITMÉTICA de seus vizinhos.')
print('  → Um pixel de ruído 255 "contamina" todos os K² vizinhos ao redor:')
print('    ex: média(255, 120, 118, 122, 119) ≈ 147 → todos ficam acinzentados.')
print('  → O pico de ruído desaparece, mas deixa uma "mancha cinza" em volta.')
print()
print('Filtro Gaussiano:')
print('  → Similar à Média, mas pesos maiores no centro (distribuição gaussiana).')
print('  → O pixel ruidoso tem peso alto → ainda influencia muito a saída.')
print()
print('A raiz do problema é que ambos são filtros LINEARES:')
print('  → Operam com MÉDIAS PONDERADAS → incluem o valor ruidoso na conta.')
print('  → Solução: filtro de MEDIANA (não-linear) — substitui pelo VALOR DO MEIO')
print('    após ordenar os K² vizinhos → ignora completamente os extremos (0 e 255).')
