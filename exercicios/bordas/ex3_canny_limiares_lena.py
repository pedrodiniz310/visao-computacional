"""
Exercício 3 — Canny com Diferentes Limiares

Utilizando o detector de Canny, defina o limiar inferior e superior com valores
muito próximos (ex: 100 e 110). Depois, afaste-os (ex: 50 e 200). Relate o que
acontece com o ruído de fundo e com a continuidade das bordas na imagem lena.png.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminho absoluto para a imagem lena.png
PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/lena.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# Converter para escala de cinza: Canny opera em 1 canal
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Suavização leve com Gaussiano 5×5 para base consistente entre os testes
# Isso remove micro-ruído da imagem antes do cálculo do gradiente
cinza_suave = cv2.GaussianBlur(cinza, (5, 5), 0)

# ── Configurações de limiares para comparação ──────────────────────────────────
# O Canny usa histérese com dois limiares:
#   T1 (limiar inferior): bordas ABAIXO são descartadas
#   T2 (limiar superior): bordas ACIMA são aceitas como bordas fortes (âncoras)
#   Bordas ENTRE T1 e T2 são aceitas SOMENTE se conectadas a uma borda forte
configs = [
    (100, 110, 'Limiares PRÓXIMOS\n(100, 110)'),      # janela de histérese pequena
    (50,  200, 'Limiares AFASTADOS\n(50, 200)'),       # janela ampla: boa razão 1:4
    (30,  100, 'Limiares INTERMEDIÁRIOS\n(30, 100)'),  # razão 1:3,3 (próxima do ideal)
]

# Aplica Canny para cada configuração de limiar
resultados = [(thr1, thr2, titulo, cv2.Canny(cinza_suave, thr1, thr2))
              for thr1, thr2, titulo in configs]

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Canny — Impacto dos Limiares nas Bordas (lena.png)', fontsize=13)

# Original em cinza para referência
axes[0, 0].imshow(cinza, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original (cinza)'); axes[0, 0].axis('off')

# Resultados do Canny com contagem de pixels de borda
for i, (thr1, thr2, titulo, canny_img) in enumerate(resultados):
    n_pixels = int(np.count_nonzero(canny_img))  # conta pixels de borda (≠ 0)
    axes[0, i + 1].imshow(canny_img, cmap='gray', vmin=0, vmax=255)
    axes[0, i + 1].set_title(f'{titulo}\nPixels de borda: {n_pixels:,}')
    axes[0, i + 1].axis('off')

# ── Histograma do gradiente para contextualizar a escolha dos limiares ─────────
# O histograma do gradiente ajuda a escolher T1 e T2 visualmente:
# T2 deve ser posicionado onde começam os gradientes fortes (bordas reais)
# T1 deve ser ~1/3 de T2 (regra prática recomendada)
sobel_x   = cv2.Sobel(cinza_suave, cv2.CV_64F, 1, 0, ksize=3)  # gradiente X
sobel_y   = cv2.Sobel(cinza_suave, cv2.CV_64F, 0, 1, ksize=3)  # gradiente Y
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)   # magnitude real: √(Gx² + Gy²)
mag_uint8 = cv2.convertScaleAbs(magnitude)      # normaliza para [0-255] uint8

axes[1, 0].hist(mag_uint8.ravel(), bins=256, range=(1, 256), color='black', density=True)
axes[1, 0].set_title('Distribuição do gradiente\n(auxilia escolha dos limiares)')
axes[1, 0].set_xlabel('Magnitude do gradiente')
axes[1, 0].set_ylabel('Freq. relativa')

# Sobreposição das bordas Canny (vermelho) na imagem original
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
for i, (_, _, titulo, canny_img) in enumerate(resultados):
    overlay = imagem.copy()
    overlay[canny_img > 0] = [0, 0, 255]  # pinta as bordas de vermelho (BGR)
    axes[1, i + 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, i + 1].set_title(f'Bordas sobrepostas\n{titulo}')
    axes[1, i + 1].axis('off')

plt.tight_layout()
plt.show()

# ── Relatório textual ─────────────────────────────────────────────────────────
print('=== Impacto dos Limiares no Algoritmo de Canny ===')
print()
print('REVISÃO DA HISTÉRESE DO CANNY:')
print('  → threshold1 (T1): limiar INFERIOR — bordas abaixo disso são descartadas.')
print('  → threshold2 (T2): limiar SUPERIOR — bordas acima disso são aceitas.')
print('  → Bordas entre T1 e T2 são aceitas APENAS se conectadas a uma borda forte.')
print()
print('─' * 60)
print('Limiares PRÓXIMOS (100, 110):')
print('─' * 60)
print('  → A janela de histérese é mínima (apenas 10 unidades).')
print('  → Resultado: muito RUÍDO de fundo + bordas fragmentadas.')
print()
print('─' * 60)
print('Limiares AFASTADOS (50, 200):')
print('─' * 60)
print('  → T2 alto (200): apenas bordas com gradiente muito forte são âncoras.')
print('  → T1 baixo (50): bordas fracas conectadas a âncoras são aceitas.')
print('  → Resultado: MENOS ruído de fundo + bordas principais MAIS CONTÍNUAS.')
print()
print('Regra prática recomendada:')
print('  → T2 / T1 ≈ 3:1  (ex: T1=50, T2=150)')
