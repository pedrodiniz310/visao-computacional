"""
Exercício 2 — Sobel vs. Canny
Compare visualmente o resultado do Sobel (magnitude combinada) com o algoritmo
de Canny da imagem objetos.jpg. Qual deles produz bordas mais finas
("esqueletizadas") e prontas para contagem de objetos?
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/objetos.jpg'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Sobel — magnitude combinada ───────────────────────────────────────────────
# cv2.Sobel(src, ddepth, dx, dy, ksize):
#   ddepth=CV_64F: resultado em float64 para preservar valores negativos
#     (um gradiente negativo significa mudança de claro para escuro)
#   dx=1, dy=0: derivada na direção X (detecta bordas verticais)
#   dx=0, dy=1: derivada na direção Y (detecta bordas horizontais)
#   ksize=3: kernel 3×3 (ksize=1 usa diferenças simples, maior = mais suave)
sobel_x = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(cinza, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude real: sqrt(Gx² + Gy²) — combina gradientes em X e Y
# cv2.convertScaleAbs(): aplica |x| e converte para uint8 (satura em 255)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_abs  = cv2.convertScaleAbs(magnitude)

# Binarizar Sobel para comparação justa com Canny
# limiar 50: pixels com magnitude > 50 são considerados bordas
_, sobel_bin = cv2.threshold(sobel_abs, 50, 255, cv2.THRESH_BINARY)

# ── Canny ───────────────────────────────────────────────────────────────────
# cv2.Canny(): detector de bordas com supressão de não-máximos + histérese
#   threshold1=50:  limiar inferior (bordas fracas abaixo disso: descartadas)
#   threshold2=150: limiar superior (bordas fortes acima disso: aceitas)
#   Bordas entre 50-150 são aceitas apenas se conectadas a bordas fortes
# Resultado: bordas FINAS (1 pixel de espessura) vs. Sobel que pode ser espesso
canny = cv2.Canny(cinza, threshold1=50, threshold2=150)

# ── Contar componentes conectados (bordas detectadas) ─────────────────────────
# cv2.connectedComponents(): rotula regiões brancas conectadas com números sequenciais
# Retorna (n_labels, mapa_de_rótulos) onde n_labels inclui o fundo (rótulo 0)
# Por isso subtraímos 1 para contar apenas os objetos reais
n_sobel, _ = cv2.connectedComponents(sobel_bin)
n_canny, _ = cv2.connectedComponents(canny)

# ── Exibição ──────────────────────────────────────────────────────────────────
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Sobel (magnitude) vs. Canny — Detecção de Bordas', fontsize=13)

axes[0, 0].imshow(imagem_rgb)
axes[0, 0].set_title('Original'); axes[0, 0].axis('off')

axes[0, 1].imshow(sobel_abs, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Sobel — Magnitude (escala cinza)'); axes[0, 1].axis('off')

axes[0, 2].imshow(canny, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('Canny (bordas finas/esqueletizadas)'); axes[0, 2].axis('off')

axes[1, 0].imshow(sobel_bin, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title(f'Sobel binarizado (thr=50)\nComponentes: {n_sobel - 1}'); axes[1, 0].axis('off')

axes[1, 1].imshow(canny, cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title(f'Canny\nComponentes: {n_canny - 1}'); axes[1, 1].axis('off')

# Sobreposição: Sobel (vermelho) e Canny (verde) na imagem original
sobreposicao = imagem.copy()
sobreposicao[sobel_bin > 0] = [0, 0, 200]   # vermelho onde Sobel detectou
sobreposicao[canny > 0]     = [0, 200, 0]   # verde onde Canny detectou
# Amarelo = ambos detectaram
ambos = cv2.bitwise_and(sobel_bin, canny)
sobreposicao[ambos > 0] = [0, 200, 200]    # amarelo = detecção em comum

axes[1, 2].imshow(cv2.cvtColor(sobreposicao, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Sobreposição\nVermelho=Sobel | Verde=Canny | Amarelo=Ambos'); axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ── Métricas de espessura ─────────────────────────────────────────────────────
pixels_sobel = int(np.count_nonzero(sobel_bin))
pixels_canny = int(np.count_nonzero(canny))

print('=== Comparação quantitativa ===')
print(f'Pixels de borda Sobel:  {pixels_sobel:,}')
print(f'Pixels de borda Canny:  {pixels_canny:,}')
print(f'Razão (Sobel/Canny):    {pixels_sobel/pixels_canny:.2f}x mais pixels no Sobel')
print()
print('=== Análise comparativa ===')
print()
print('Sobel (magnitude combinada):')
print('  → Gera bordas ESPESSAS (gradiente de intensidade gera faixas largas).')
print('  → Cada borda real produz múltiplos pixels acesos em torno da transição.')
print('  → Não há supressão de não-máximos: a borda "engorda".')
print('  → Requer binarização manual com um limiar definido pelo usuário.')
print('  → Sensível a ruído: qualquer variação de intensidade gera uma "borda".')
print('  → Resultado: bordas grossas, difíceis de usar diretamente para contagem.')
print()
print('Canny:')
print('  → Gera bordas FINAS, de espessura 1 pixel (esqueletizadas).')
print('  → Etapas internas: suavização Gaussiana → gradiente → supressão de')
print('    não-máximos → histérese com dois limiares.')
print('  → A supressão de não-máximos mantém apenas o pixel de maior gradiente')
print('    ao longo da direção perpendicular à borda.')
print('  → A histérese conecta bordas fracas a bordas fortes, eliminando ruído.')
print('  → Resultado: bordas finas, contínuas e prontas para contagem de objetos.')
print()
print('CONCLUSÃO: o Canny produz bordas mais finas e prontas para contagem.')
print('O Sobel é mais adequado para realce visual ou como entrada para outros')
print('algoritmos (não para segmentação direta de objetos).')
