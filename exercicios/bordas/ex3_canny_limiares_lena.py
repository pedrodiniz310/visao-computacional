"""
Exercício 3 — Canny com Diferentes Limiares
Utilizando o detector de Canny, defina o limiar inferior e superior com valores
muito próximos (ex: 100 e 110). Depois, afaste-os (ex: 50 e 200). Relate o que
acontece com o ruído de fundo e com a continuidade das bordas na imagem lena.png.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = '../../imagens/lena.png'

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
# Suavização leve para base consistente
cinza_suave = cv2.GaussianBlur(cinza, (5, 5), 0)

# ── Configurações de limiares ─────────────────────────────────────────────────
configs = [
    (100, 110, 'Limiares PRÓXIMOS\n(100, 110)'),
    (50,  200, 'Limiares AFASTADOS\n(50, 200)'),
    (30,  100, 'Limiares INTERMEDIÁRIOS\n(30, 100)'),
]

resultados = [(thr1, thr2, titulo, cv2.Canny(cinza_suave, thr1, thr2))
              for thr1, thr2, titulo in configs]

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Canny — Impacto dos Limiares nas Bordas (lena.png)', fontsize=13)

axes[0, 0].imshow(cinza, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original (cinza)'); axes[0, 0].axis('off')

for i, (thr1, thr2, titulo, canny_img) in enumerate(resultados):
    n_pixels = int(np.count_nonzero(canny_img))
    axes[0, i + 1].imshow(canny_img, cmap='gray', vmin=0, vmax=255)
    axes[0, i + 1].set_title(f'{titulo}\nPixels de borda: {n_pixels:,}')
    axes[0, i + 1].axis('off')

# Histograma do gradiente para contextualizar a escolha dos limiares
sobel_x = cv2.Sobel(cinza_suave, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(cinza_suave, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
mag_uint8 = cv2.convertScaleAbs(magnitude)

axes[1, 0].hist(mag_uint8.ravel(), bins=256, range=(1, 256), color='black', density=True)
axes[1, 0].set_title('Distribuição do gradiente\n(auxilia escolha dos limiares)')
axes[1, 0].set_xlabel('Magnitude do gradiente')
axes[1, 0].set_ylabel('Freq. relativa')

# Sobreposição de cada resultado na imagem original
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
for i, (_, _, titulo, canny_img) in enumerate(resultados):
    overlay = imagem.copy()
    overlay[canny_img > 0] = [0, 0, 255]  # vermelho sobre bordas
    axes[1, i + 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, i + 1].set_title(f'Bordas sobrepostas\n{titulo}')
    axes[1, i + 1].axis('off')

plt.tight_layout()
plt.show()

# ── Relatório ─────────────────────────────────────────────────────────────────
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
print('  → Bordas fracas são aceitas facilmente (T1 = 100 é alto).')
print('  → Resultado: muitas bordas REAIS aparecem, mas também muito RUÍDO de fundo.')
print('  → A continuidade das bordas pode ser fragmentada pois T1 alto exige')
print('    que toda a cadeia tenha gradiente > 100.')
print()
print('─' * 60)
print('Limiares AFASTADOS (50, 200):')
print('─' * 60)
print('  → T2 alto (200): apenas bordas com gradiente muito forte são ancoras.')
print('  → T1 baixo (50): bordas fracas conectadas a ancoras são aceitas.')
print('  → Resultado: MENOS ruído de fundo (bordas espúrias isoladas descartadas)')
print('    e bordas principais MAIS CONTÍNUAS (conexão via limiares baixo).')
print('  → Trade-off: detalhes muito suaves podem não aparecer.')
print()
print('Regra prática recomendada:')
print('  → T2 / T1 ≈ 3:1  (ex: T1=50, T2=150)')
print('  → Ou usar o desvio-padrão do gradiente para definir automaticamente:')
print('    T2 = média + 2*desvio,  T1 = média - 2*desvio (após suavização Gaussiana)')
