"""
Exercício 5 — Ganho e Brilho vs. Equalização Automática
Crie um script que multiplique todos os pixels da imagem por um fator constante
(ganho) e some um valor fixo (brilho). Compare o resultado visual com a
equalização automática do OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = '../../imagens/frutas_escura.png'

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Transformação linear: f(x) = alpha * x + beta ────────────────────────────
# alpha: ganho (contraste)  — valores > 1 aumentam o contraste
# beta:  brilho (offset)    — valores positivos clareiam
alpha = 1.8   # ganho de contraste
beta  = 30    # incremento de brilho

# convertScaleAbs aplica a fórmula e satura em [0, 255]
ajustada = cv2.convertScaleAbs(cinza, alpha=alpha, beta=beta)

# ── Equalização global automática ────────────────────────────────────────────
equalizada = cv2.equalizeHist(cinza)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(f'Ganho/Brilho (α={alpha}, β={beta}) vs. Equalização Automática', fontsize=13)

# Linha 1: imagens
axes[0, 0].imshow(cinza,     cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original');   axes[0, 0].axis('off')

axes[0, 1].imshow(ajustada,  cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title(f'Ganho/Brilho (α={alpha}, β={beta})'); axes[0, 1].axis('off')

axes[0, 2].imshow(equalizada, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('Equalização Automática'); axes[0, 2].axis('off')

# Linha 2: histogramas
hist_orig = cv2.calcHist([cinza],     [0], None, [256], [0, 256])
hist_aj   = cv2.calcHist([ajustada],  [0], None, [256], [0, 256])
hist_eq   = cv2.calcHist([equalizada],[0], None, [256], [0, 256])

axes[1, 0].plot(hist_orig, color='black')
axes[1, 0].set_title('Histograma — Original'); axes[1, 0].set_xlim([0, 256])

axes[1, 1].plot(hist_aj, color='steelblue')
axes[1, 1].set_title('Histograma — Ganho/Brilho'); axes[1, 1].set_xlim([0, 256])

axes[1, 2].plot(hist_eq, color='darkorange')
axes[1, 2].set_title('Histograma — Equalizado'); axes[1, 2].set_xlim([0, 256])

plt.tight_layout()
plt.show()

# ── Estatísticas ──────────────────────────────────────────────────────────────
print('=== Comparação estatística ===')
for nome, img in [('Original', cinza), ('Ganho/Brilho', ajustada), ('Equalizada', equalizada)]:
    print(f'\n{nome}:')
    print(f'  Min:  {int(img.min())}  |  Max: {int(img.max())}  |  '
          f'Média: {img.mean():.1f}  |  Desvio-padrão: {img.std():.1f}')

print()
print('=== Comparação conceitual ===')
print()
print('Transformação linear (Ganho e Brilho):')
print('  → Aplica a fórmula f(x) = α·x + β a cada pixel.')
print('  → α > 1 "estica" o histograma aumentando o contraste.')
print('  → β > 0 desloca todo o histograma para a direita (clareia).')
print('  → É uma transformação GLOBAL E UNIFORME: todas as regiões são')
print('    afetadas da mesma forma, independente do conteúdo local.')
print('  → Pixels nas extremidades (0 ou 255) são saturados ("cortados").')
print('  → Requer ajuste MANUAL de α e β pelo usuário.')
print()
print('Equalização Automática:')
print('  → Distribui automaticamente os pixels por toda a faixa [0, 255].')
print('  → O histograma resultante tende a ser mais uniforme (flat).')
print('  → Não requer parâmetros manuais: adapta-se ao conteúdo da imagem.')
print('  → Pode superexpor regiões já claras ou subexpor detalhes.')
print()
print('Conclusão: a equalização automática é mais adaptativa, mas pode produzir')
print('resultados com "halos" ou artefatos. O ajuste manual oferece mais controle,')
print('porém exige conhecimento prévio do conteúdo da imagem.')
