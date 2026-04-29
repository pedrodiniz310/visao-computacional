"""
Exercício 3 — Binarização: Valor Fixo vs. Otsu

Realize a binarização na imagem lena.png usando um valor fixo (127) e,
em seguida, utilize o método de Otsu. Descreva em quais situações o cálculo
automático de Otsu se mostra superior ao valor fixo.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Caminho absoluto para a imagem padrão de teste lena.png
PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/lena.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
# cv2.IMREAD_GRAYSCALE: necessário pois cv2.threshold() opera em 1 canal
imagem = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Binarização com valor fixo ────────────────────────────────────────────────
# cv2.threshold(src, limiar, maxVal, tipo) → (limiar_retornado, imagem_binária)
# THRESH_BINARY: pixel > limiar → maxVal(255); pixel ≤ limiar → 0
# O limiar 127 corresponde ao meio da escala [0-255], mas pode ser inadequado
# para imagens com iluminação incomum (muito claras ou muito escuras).
LIMIAR_FIXO = 127
_, bin_fixa = cv2.threshold(imagem, LIMIAR_FIXO, 255, cv2.THRESH_BINARY)

# ── Binarização com Otsu ──────────────────────────────────────────────────────
# THRESH_OTSU: ignora o limiar informado (0) e CALCULA automaticamente o melhor
# limiar para a imagem. O algoritmo de Otsu maximiza a variância ENTRE as duas
# classes (fundo e objeto), equivalente a encontrar o "vale" do histograma bimodal.
# O primeiro retorno contém o limiar calculado (útil para exibir e comparar).
limiar_otsu, bin_otsu = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ── Calcular histograma para visualização ────────────────────────────────────
# cv2.calcHist([imgs], [canais], máscara, [nBins], [range])
# 256 bins → um bin por nível de intensidade (0 a 255)
hist = cv2.calcHist([imagem], [0], None, [256], [0, 256])

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Binarização: Limiar Fixo vs. Método de Otsu', fontsize=13)

# Linha superior: comparação visual das binarizações
axes[0, 0].imshow(imagem,   cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original (cinza)'); axes[0, 0].axis('off')

axes[0, 1].imshow(bin_fixa, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title(f'Limiar Fixo = {LIMIAR_FIXO}'); axes[0, 1].axis('off')

axes[0, 2].imshow(bin_otsu, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title(f'Método de Otsu (limiar automático = {int(limiar_otsu)})'); axes[0, 2].axis('off')

# Histograma com as duas linhas de limiar sobrepostas
axes[1, 0].plot(hist, color='black', linewidth=1)
# axvline(): traça linha vertical no valor x indicado
axes[1, 0].axvline(x=LIMIAR_FIXO, color='red',   linestyle='--', label=f'Fixo ({LIMIAR_FIXO})')
axes[1, 0].axvline(x=limiar_otsu, color='green', linestyle='--', label=f'Otsu ({int(limiar_otsu)})')
axes[1, 0].set_title('Histograma com limiares')
axes[1, 0].set_xlim([0, 256])
axes[1, 0].legend()

# Mapa de diferenças: cv2.absdiff() calcula |bin_fixa - bin_otsu| pixel a pixel
# Pixels brancos = posições onde os dois métodos DISCORDAM
diferenca = cv2.absdiff(bin_fixa, bin_otsu)
axes[1, 1].imshow(diferenca, cmap='hot', vmin=0, vmax=255)
axes[1, 1].set_title('Diferença entre as binarizações\n(pixels divergentes em branco)'); axes[1, 1].axis('off')

# Painel de estatísticas
diferenca_pct = (np.count_nonzero(diferenca) / diferenca.size) * 100
axes[1, 2].axis('off')
texto = (
    f'Limiar Fixo:  {LIMIAR_FIXO}\n'
    f'Limiar Otsu:  {int(limiar_otsu)}\n\n'
    f'Pixels divergentes: {diferenca_pct:.1f}%\n\n'
    f'Pixels brancos (fixo): {int(np.sum(bin_fixa == 255)):,}\n'
    f'Pixels brancos (Otsu): {int(np.sum(bin_otsu == 255)):,}'
)
axes[1, 2].text(0.1, 0.5, texto, transform=axes[1, 2].transAxes,
                fontsize=12, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 2].set_title('Estatísticas')

plt.tight_layout()
plt.show()

# ── Análise ───────────────────────────────────────────────────────────────────
print('=== Quando o Otsu é SUPERIOR ao limiar fixo? ===')
print()
print('1. IMAGENS COM ILUMINAÇÃO VARIÁVEL:')
print('   → O limiar fixo pode funcionar para uma imagem e falhar em outra.')
print('   → O Otsu recalcula o limiar ideal para CADA imagem individualmente.')
print()
print('2. IMAGENS BIMODAIS (dois grupos claros no histograma):')
print('   → O Otsu encontra o vale entre os dois picos do histograma.')
print('   → O limiar fixo pode cair fora do vale ideal.')
print()
print('3. SISTEMAS AUTOMATIZADOS (câmera sem controle de iluminação):')
print('   → Em esteiras industriais, robótica, câmeras externas etc.')
print('   → O Otsu adapta-se automaticamente sem intervenção humana.')
print()
print('4. IMAGENS COM HISTOGRAMA DESLOCADO:')
print('   → Se a imagem for muito clara ou escura, o fixo erra sistematicamente.')
print('   → O Otsu encontra a separação correta independente do nível médio.')
