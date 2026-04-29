"""
Exercício 2 — Filtro de Mediana com Kernel Gradual
Aplique o filtro de Mediana na imagem sal_e_pimenta.png. Aumente o tamanho
do kernel gradualmente (3, 5, 9). O que acontece com os detalhes finos da
imagem conforme o kernel cresce?
"""

import cv2
import matplotlib.pyplot as plt

PATH = '../../imagens/sal_e_pimenta.png'

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Aplicar Mediana com diferentes kernels ────────────────────────────────────
kernels    = [3, 5, 9]
resultados = {k: cv2.medianBlur(imagem, k) for k in kernels}

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Filtro de Mediana — Efeito do Tamanho do Kernel', fontsize=13)

axes[0, 0].imshow(imagem,           cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original — Ruído Sal e Pimenta'); axes[0, 0].axis('off')

for ax, k in zip([axes[0, 1], axes[1, 0], axes[1, 1]], kernels):
    ax.imshow(resultados[k], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Mediana kernel {k}x{k}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# ── Comparação lado a lado de histogramas ─────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
fig2.suptitle('Histogramas — Original e Medianas', fontsize=12)

todas = [('Original', imagem)] + [(f'k={k}', resultados[k]) for k in kernels]
for ax, (titulo, img) in zip(axes2, todas):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    ax.plot(hist, color='black')
    ax.set_title(titulo)
    ax.set_xlim([0, 256])

plt.tight_layout()
plt.show()

# ── Análise ───────────────────────────────────────────────────────────────────
print('=== O que acontece com os detalhes conforme o kernel cresce? ===')
print()
print('Filtro de Mediana 3x3:')
print('  → Remove a maioria dos pixels ruidosos isolados.')
print('  → Bordas e detalhes finos são bem preservados.')
print('  → Ideal para ruído espalhado e pouco denso.')
print('  → Efeito visual: imagem limpa, muito próxima da original sem ruído.')
print()
print('Filtro de Mediana 5x5:')
print('  → Remove eficientemente o ruído, inclusive ruídos em grupos de 2-3 pixels.')
print('  → Começa a suavizar levemente os detalhes muito finos (texturas, linhas finas).')
print('  → Bordas principais ainda são bem preservadas.')
print('  → Boa escolha quando o ruído é denso.')
print()
print('Filtro de Mediana 9x9:')
print('  → Elimina virtualmente todo o ruído, inclusive grãos grandes.')
print('  → DESTRÓI detalhes finos: texturas, linhas finas e bordas fracas desaparecem.')
print('  → A imagem fica com aparência "borrachosa" ou de pintura.')
print('  → Objetos pequenos podem ser completamente eliminados.')
print()
print('Conclusão:')
print('  → Existe um trade-off entre remoção de ruído e preservação de detalhes.')
print('  → Kernels pequenos preservam detalhes mas podem não remover todo o ruído.')
print('  → Kernels grandes removem mais ruído mas borram detalhes finos.')
print('  → A escolha ideal depende da densidade do ruído e da importância dos detalhes.')
