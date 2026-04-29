"""
Exercício 2 — Limiarização Global vs. Adaptativa no Sudoku
Utilize a imagem sudoku.png. Aplique a limiarização global e a limiarização
adaptativa e compare qual método preservou melhor a legibilidade do texto.
Por que isso ocorre?
"""

import os
import cv2
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/sudoku.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Suavização leve ───────────────────────────────────────────────────────────
suave = cv2.GaussianBlur(cinza, (5, 5), 0)

# ── Limiarização Global (Otsu) ────────────────────────────────────────────────
# cv2.threshold(src, limiar, maxVal, tipo):
#   THRESH_BINARY: pixel > limiar → maxVal(255); senao → 0
#   THRESH_OTSU: ignora o limiar informado (0) e calcula automaticamente
#     o melhor limiar maximizando a variância ENTRE os dois grupos (fundo/objeto)
#   Primeiro retorno = limiar calculado pelo Otsu
limiar_otsu, global_otsu = cv2.threshold(suave, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Limiar fixo como referência (128 = meio da escala 0-255)
_, global_fixo = cv2.threshold(suave, 128, 255, cv2.THRESH_BINARY)

# ── Limiarização Adaptativa ─────────────────────────────────────────────────────
# cv2.adaptiveThreshold(src, maxVal, metodo, tipo, blockSize, C):
#   metodo ADAPTIVE_THRESH_MEAN_C: limiar = MEDIA da vizinhança blockSize×blockSize minus C
#   metodo ADAPTIVE_THRESH_GAUSSIAN_C: limiar = MÉDIA PONDERADA (gaussiana) minus C
#   blockSize=11: janela 11×11 pixels para calcular o limiar local (deve ser ímpar)
#   C=2: constante subtraída da média (ajusta o limiar para baixo/cima)
# VANTAGEM sobre global: compensa iluminação não-uniforme (documento fotografado na diagonal)
adaptativa_mean = cv2.adaptiveThreshold(suave, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

# ADAPTIVE_THRESH_GAUSSIAN_C: usa média ponderada gaussiana (mais suave)
adaptativa_gauss = cv2.adaptiveThreshold(suave, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('Limiarização Global vs. Adaptativa — sudoku.png', fontsize=13)

axes[0, 0].imshow(cinza, cmap='gray')
axes[0, 0].set_title('Original (cinza)'); axes[0, 0].axis('off')

axes[0, 1].imshow(global_fixo, cmap='gray')
axes[0, 1].set_title('Global Fixo (limiar=128)\nPode perder dígitos escuros'); axes[0, 1].axis('off')

axes[0, 2].imshow(global_otsu, cmap='gray')
axes[0, 2].set_title(f'Global Otsu (limiar={int(limiar_otsu)})\nAdapta ao histograma global'); axes[0, 2].axis('off')

axes[1, 0].imshow(adaptativa_mean, cmap='gray')
axes[1, 0].set_title('Adaptativa — Média (11x11, C=2)\nLegibilidade local melhorada'); axes[1, 0].axis('off')

axes[1, 1].imshow(adaptativa_gauss, cmap='gray')
axes[1, 1].set_title('Adaptativa — Gaussiana (11x11, C=2)\nMelhor para texto cursivo/irregular'); axes[1, 1].axis('off')

# Recorte central para demonstrar legibilidade
h, w = cinza.shape
cy, cx = h // 2, w // 2
tam = min(h, w) // 4
recortes = [
    ('Global Otsu',   global_otsu),
    ('Adapt. Gauss',  adaptativa_gauss),
]
texto_recorte = 'Zoom região central\n'
for nome, img in recortes:
    texto_recorte += f'[{nome}] '

recorte_orig  = cinza[cy-tam:cy+tam, cx-tam:cx+tam]
recorte_otsu  = global_otsu[cy-tam:cy+tam, cx-tam:cx+tam]
recorte_adapt = adaptativa_gauss[cy-tam:cy+tam, cx-tam:cx+tam]
import numpy as np
comparacao = np.hstack([
    cv2.resize(recorte_orig,  (200, 200)),
    cv2.resize(recorte_otsu,  (200, 200)),
    cv2.resize(recorte_adapt, (200, 200)),
])
axes[1, 2].imshow(comparacao, cmap='gray')
axes[1, 2].set_title('Zoom: Original | Otsu | Adapt. Gauss\n(Adaptativa preserva melhor os dígitos)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ── Análise ───────────────────────────────────────────────────────────────────
print('=== Comparação de Métodos de Limiarização — sudoku.png ===')
print()
print(f'Limiar Global Fixo:    128')
print(f'Limiar Global Otsu:    {int(limiar_otsu)}')
print()
print('LIMIARIZAÇÃO GLOBAL:')
print('  → Aplica UM ÚNICO limiar para toda a imagem.')
print('  → Funciona bem quando a iluminação é UNIFORME em toda a cena.')
print('  → No sudoku, onde há variações de iluminação (um lado mais claro,')
print('    outro mais escuro), o limiar global pode:')
print('    - Apagar dígitos nas áreas escuras (abaixo do limiar)')
print('    - Deixar manchas nas áreas muito claras (acima do limiar)')
print()
print('LIMIARIZAÇÃO ADAPTATIVA:')
print('  → Divide a imagem em blocos e calcula um limiar DIFERENTE para cada bloco.')
print('  → O limiar de cada pixel é: média_local(vizinhos) - C')
print('  → Independente da iluminação global, o texto local é sempre detectado.')
print('  → Resultado: dígitos legíveis em TODAS as regiões da imagem.')
print()
print('POR QUE A ADAPTATIVA É SUPERIOR NO SUDOKU?')
print('  → A imagem do sudoku frequentemente tem iluminação não-uniforme:')
print('    sombras nas bordas, reflexo no centro, fotografia inclinada, etc.')
print('  → A limiarização global trata todos os pixels com a mesma régua.')
print('  → A adaptativa "se calibra" localmente, como um humano que ajusta')
print('    o foco ao olhar para diferentes partes da imagem.')
print()
print('Parâmetros-chave da adaptativa:')
print('  → blockSize (11): tamanho da janela local (deve ser ímpar)')
print('     Maior = considera contexto mais amplo | Menor = mais sensível ao detalhe')
print('  → C (2): constante subtraída da média para ajustar a sensibilidade')
print('     Maior = menos pixels são considerados texto | Menor = mais pixels')
