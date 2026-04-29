"""
Exercício 4 — Morfologia para Tratar Engrenagem
Utilize operações morfológicas para tratar a imagem engrenagem.png,
removendo imperfeições causadas na binarização.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/engrenagem.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Pré-processamento: suavização para reduzir ruído antes de binarizar ───────
filtrada = cv2.medianBlur(cinza, 5)

# ── Binarização com Otsu ──────────────────────────────────────────────────────
limiar, binaria = cv2.threshold(filtrada, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

print(f'Limiar de Otsu calculado: {int(limiar)}')

# ── Elemento estruturante ─────────────────────────────────────────────────────
# cv2.getStructuringElement(forma, tamanho):
#   MORPH_RECT:    retângulo — mais agressivo nas bordas retas
#   MORPH_ELLIPSE: elipse — ideal para formas circulares como engrenagens
#   MORPH_CROSS:   cruz — preserva melhor a forma do objeto
#   (5, 5): kernel 5×5 — kernels maiores = efeito morfológico mais amplo
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ── Operações morfológicas ──────────────────────────────────────────────────────

# 1. ABERTURA (Erosão → Dilatação):
#    Erosão remove pixels brancos nas bordas de objetos brancos.
#    Dilatação restaura o tamanho do objeto principal.
#    Efeito líquido: elimina objetos brancos MENORES que o kernel (ruído)
#    sem alterar objetos maiores que o kernel.
abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

# 2. FECHAMENTO (Dilatação → Erosão):
#    Dilatação expande pixels brancos, preenchendo buracos e lacunas pequenas.
#    Erosão restaura o tamanho original.
#    Efeito líquido: preenche buracos MENORES que o kernel dentro do objeto.
fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel)

# 3. Pipeline completo: abertura → fechamento → abertura (para casos difíceis)
# kernel_grande (7×7): remove imperfeições maiores que não foram tratadas com 5×5
kernel_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
abertura2  = cv2.morphologyEx(fechamento, cv2.MORPH_OPEN,  kernel_grande)
resultado  = cv2.morphologyEx(abertura2,  cv2.MORPH_CLOSE, kernel_grande)

# ── Exibição: pipeline de processamento ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Pipeline Morfológico — Tratamento da Engrenagem', fontsize=13)

imagens = [
    (cinza,       'Cinza (original)'),
    (binaria,     f'Binarizada (Otsu, limiar={int(limiar)})'),
    (abertura,    'Abertura (kernel 5x5)\nRemove ruído externo'),
    (fechamento,  'Fechamento (kernel 5x5)\nPreenche imperfeições'),
    (resultado,   'Pipeline completo\nAbertura → Fechamento (x2)'),
]

for ax, (img, titulo) in zip(axes.flat[:5], imagens):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(titulo)
    ax.axis('off')

# Comparação: binária original vs resultado final sobrepostos em cores
comparacao = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
# Pintar regiões removidas (eram brancas na binária, pretas no resultado) em vermelho
ruido_removido = cv2.bitwise_and(binaria, cv2.bitwise_not(resultado))
comparacao[ruido_removido > 0] = [0, 0, 220]   # vermelho = ruído removido
# Pintar regiões preenchidas (eram pretas na binária, brancas no resultado) em verde
buracos_preenchidos = cv2.bitwise_and(cv2.bitwise_not(binaria), resultado)
comparacao[buracos_preenchidos > 0] = [0, 220, 0]  # verde = preenchido

axes[1, 2].imshow(cv2.cvtColor(comparacao, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Mapa de mudanças\nVermelho=ruído removido | Verde=buracos preenchidos')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ── Estatísticas ──────────────────────────────────────────────────────────────
pixels_ruido    = int(np.count_nonzero(ruido_removido))
pixels_preench  = int(np.count_nonzero(buracos_preenchidos))
pixels_binaria  = int(np.count_nonzero(binaria))
pixels_resultado = int(np.count_nonzero(resultado))

print()
print('=== Relatório de limpeza morfológica ===')
print(f'Pixels brancos na binarização original: {pixels_binaria:,}')
print(f'Pixels brancos no resultado final:      {pixels_resultado:,}')
print(f'Pixels de ruído removidos:              {pixels_ruido:,}')
print(f'Pixels de buracos preenchidos:          {pixels_preench:,}')
print()
print('=== Explicação das operações ===')
print()
print('ABERTURA (MORPH_OPEN = Erosão → Dilatação):')
print('  → A erosão encolhe as regiões brancas, eliminando pixels isolados')
print('    e pequenas "ilhas" de ruído que são menores que o kernel.')
print('  → A dilatação seguinte restaura o tamanho original dos objetos grandes,')
print('    mas os pequenos ruídos já foram eliminados pela erosão.')
print('  → Resultado: remove imperfeições EXTERNAS ao objeto.')
print()
print('FECHAMENTO (MORPH_CLOSE = Dilatação → Erosão):')
print('  → A dilatação expande as regiões brancas, preenchendo buracos internos')
print('    e lacunas pequenas dentro do objeto.')
print('  → A erosão seguinte restaura o contorno externo original do objeto,')
print('    mantendo os buracos preenchidos.')
print('  → Resultado: preenche imperfeições INTERNAS ao objeto.')
print()
print('Ordem recomendada: Abertura primeiro (elimina ruído externo),')
print('depois Fechamento (preenche buracos). Assim evita-se que o')
print('Fechamento amplifique ruídos que ainda não foram removidos.')
