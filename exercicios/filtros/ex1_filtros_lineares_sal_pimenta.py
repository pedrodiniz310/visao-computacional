"""
Exercício 1 — Filtros Lineares no Ruído Sal e Pimenta
Carregue a imagem sal_e_pimenta.png e aplique os filtros de Média e Gaussiano
com o mesmo tamanho de kernel (7x7). Explique por que esses filtros lineares
apenas "espalham" o ruído em vez de removê-lo completamente.
"""

import cv2
import matplotlib.pyplot as plt

PATH = '../../imagens/sal_e_pimenta.png'

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

K = 7  # tamanho do kernel (deve ser ímpar)

# ── Aplicar filtros ───────────────────────────────────────────────────────────
media     = cv2.blur(imagem,        (K, K))
gaussiano = cv2.GaussianBlur(imagem, (K, K), 0)

# ── Exibição ──────────────────────────────────────────────────────────────────
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

# ── Explicação ────────────────────────────────────────────────────────────────
print('=== Por que filtros lineares apenas "espalham" o ruído Sal e Pimenta? ===')
print()
print('O ruído Sal e Pimenta é caracterizado por pixels com valores EXTREMOS:')
print('  → Sal:    pixels com valor 255 (branco puro)')
print('  → Pimenta: pixels com valor 0  (preto puro)')
print()
print('Filtro de Média:')
print('  → Substitui cada pixel pela MÉDIA ARITMÉTICA de seus vizinhos.')
print('  → Um pixel de ruído com valor 255 "contamina" todos os vizinhos:')
print('    ex: média(255, 120, 118, 122, 119) ≈ 147 → todos ficam acinzentados.')
print('  → O pico de ruído desaparece, mas deixa uma "mancha cinza" em volta.')
print('  → Resultado: o ruído é DILUÍDO e ESPALHADO pela vizinhança.')
print()
print('Filtro Gaussiano:')
print('  → Similar à Média, mas com pesos maiores no centro (distribuição Gaussiana).')
print('  → O pixel ruidoso tem peso alto na conta → ainda influencia muito a saída.')
print('  → Atenua um pouco mais que a Média simples, mas ainda não elimina.')
print()
print('A raiz do problema é que ambos são filtros LINEARES:')
print('  → Incluem TODOS os pixels na média, incluindo os ruidosos.')
print('  → Não conseguem distinguir pixels de ruído dos pixels legítimos.')
print()
print('Solução: usar o Filtro de MEDIANA (não-linear):')
print('  → Ordena os pixels da vizinhança e escolhe o VALOR DO MEIO.')
print('  → Pixels extremos (0 ou 255) são sempre excluídos da seleção final.')
print('  → Remove o ruído preservando as bordas da imagem.')
