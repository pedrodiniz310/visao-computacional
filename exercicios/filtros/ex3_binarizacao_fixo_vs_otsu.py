"""
Exercício 3 — Binarização: Valor Fixo vs. Otsu
Realize a binarização na imagem lena.png usando um valor fixo (127) e,
em seguida, utilize o método de Otsu. Descreva em quais situações o cálculo
automático de Otsu se mostra superior ao valor fixo.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

PATH = '../../imagens/lena.png'

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Binarização com valor fixo ────────────────────────────────────────────────
LIMIAR_FIXO = 127
_, bin_fixa = cv2.threshold(imagem, LIMIAR_FIXO, 255, cv2.THRESH_BINARY)

# ── Binarização com Otsu ──────────────────────────────────────────────────────
limiar_otsu, bin_otsu = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ── Histograma com o limiar marcado ───────────────────────────────────────────
hist = cv2.calcHist([imagem], [0], None, [256], [0, 256])

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Binarização: Limiar Fixo vs. Método de Otsu', fontsize=13)

axes[0, 0].imshow(imagem, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original (cinza)'); axes[0, 0].axis('off')

axes[0, 1].imshow(bin_fixa, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title(f'Limiar Fixo = {LIMIAR_FIXO}'); axes[0, 1].axis('off')

axes[0, 2].imshow(bin_otsu, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title(f'Método de Otsu (limiar automático = {int(limiar_otsu)})'); axes[0, 2].axis('off')

# Histograma com limiares marcados
axes[1, 0].plot(hist, color='black', linewidth=1)
axes[1, 0].axvline(x=LIMIAR_FIXO,  color='red',    linestyle='--', label=f'Fixo ({LIMIAR_FIXO})')
axes[1, 0].axvline(x=limiar_otsu,  color='green',  linestyle='--', label=f'Otsu ({int(limiar_otsu)})')
axes[1, 0].set_title('Histograma com limiares')
axes[1, 0].set_xlim([0, 256])
axes[1, 0].legend()

# Diferença entre as binarizações
diferenca = cv2.absdiff(bin_fixa, bin_otsu)
axes[1, 1].imshow(diferenca, cmap='hot', vmin=0, vmax=255)
axes[1, 1].set_title('Diferença entre as binarizações\n(pixels divergentes em branco)'); axes[1, 1].axis('off')

# Estatísticas
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
print('=== Binarização Fixo vs. Otsu ===')
print(f'Limiar fixo:  {LIMIAR_FIXO}')
print(f'Limiar Otsu:  {int(limiar_otsu)}')
print()
print('=== Quando o Otsu é SUPERIOR ao limiar fixo? ===')
print()
print('1. IMAGENS COM ILUMINAÇÃO VARIÁVEL:')
print('   → O limiar fixo pode funcionar para uma imagem, mas falhar completamente')
print('     em outra da mesma cena com iluminação diferente.')
print('   → O Otsu recalcula o limiar ideal para CADA imagem individualmente.')
print()
print('2. IMAGENS BIMODAIS (com dois grupos claros de pixels):')
print('   → O Otsu analisa o histograma e encontra o vale entre os dois picos.')
print('   → Maximiza a variância ENTRE classes (fundo e objeto).')
print('   → O limiar fixo pode cair fora do vale ideal.')
print()
print('3. SISTEMAS AUTOMATIZADOS (câmera sem controle de iluminação):')
print('   → Em esteiras industriais, robótica ou monitoramento ao ar livre,')
print('     a iluminação muda constantemente.')
print('   → O Otsu adapta-se automaticamente sem intervenção humana.')
print()
print('4. IMAGENS COM HISTOGRAMA DESLOCADO:')
print('   → Se a imagem for predominantemente clara ou escura, o limiar fixo')
print('     pode binarizar quase tudo como branco ou preto.')
print('   → O Otsu encontra o ponto de separação correto independente do nível médio.')
print()
print('Limitação do Otsu:')
print('   → Funciona mal em imagens com iluminação NÃO UNIFORME (gradientes).')
print('   → Nesse caso, a limiarização adaptativa (cv2.adaptiveThreshold) é superior.')
