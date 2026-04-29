"""
Exercício 3 — Equalização no Canal V (HSV)
Converta uma imagem BGR para HSV, aplique a equalização de histograma apenas
no canal V (Value) e retorne para BGR. Explique por que não devemos equalizar
os canais R, G e B separadamente.
"""

import cv2
import matplotlib.pyplot as plt

PATH = '../../imagens/frutas.png'

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem_bgr = cv2.imread(PATH)
if imagem_bgr is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Conversão BGR → HSV ───────────────────────────────────────────────────────
imagem_hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

# Separar os canais H (matiz), S (saturação) e V (valor/brilho)
h, s, v = cv2.split(imagem_hsv)

# ── Equalizar apenas o canal V ────────────────────────────────────────────────
v_equalizado = cv2.equalizeHist(v)

# Remontar a imagem HSV com V equalizado
hsv_equalizado = cv2.merge([h, s, v_equalizado])

# ── Converter de volta para BGR ───────────────────────────────────────────────
resultado_bgr = cv2.cvtColor(hsv_equalizado, cv2.COLOR_HSV2BGR)

# ── Exibição ──────────────────────────────────────────────────────────────────
imagem_rgb   = cv2.cvtColor(imagem_bgr,   cv2.COLOR_BGR2RGB)
resultado_rgb = cv2.cvtColor(resultado_bgr, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Equalização no Canal V do espaço HSV', fontsize=13)

axes[0].imshow(imagem_rgb)
axes[0].set_title('Original (BGR)')
axes[0].axis('off')

axes[1].imshow(v, cmap='gray')
axes[1].set_title('Canal V — Original')
axes[1].axis('off')

axes[2].imshow(resultado_rgb)
axes[2].set_title('Resultado: V equalizado → BGR')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# ── Histogramas do canal V ────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
fig2.suptitle('Histograma do Canal V — Original vs. Equalizado', fontsize=13)

hist_v     = cv2.calcHist([v],           [0], None, [256], [0, 256])
hist_v_eq  = cv2.calcHist([v_equalizado],[0], None, [256], [0, 256])

axes2[0].plot(hist_v, color='gray')
axes2[0].set_title('V — Original')
axes2[0].set_xlim([0, 256])

axes2[1].plot(hist_v_eq, color='orange')
axes2[1].set_title('V — Equalizado')
axes2[1].set_xlim([0, 256])

plt.tight_layout()
plt.show()

# ── Explicação ────────────────────────────────────────────────────────────────
print('=== Por que não devemos equalizar R, G e B separadamente? ===')
print()
print('No espaço RGB, os canais R, G e B estão FORTEMENTE ACOPLADOS:')
print('a percepção de cor e de brilho dependem da COMBINAÇÃO dos três canais.')
print()
print('Quando equalizamos cada canal individualmente:')
print('  → Cada canal redistribui sua intensidade de forma independente.')
print('  → O balanço cromático entre R, G e B é destruído.')
print('  → A cor resultante não corresponde à cor real do objeto.')
print()
print('A solução correta é separar BRILHO de COR usando o espaço HSV:')
print('  → H (Hue/Matiz): define a cor pura. NÃO deve ser alterado.')
print('  → S (Saturation): define a vivacidade da cor. NÃO deve ser alterado.')
print('  → V (Value/Brilho): define a luminosidade. É AQUI que equalizamos.')
print()
print('Assim, apenas o contraste de brilho é melhorado, preservando as')
print('cores originais da cena sem distorções cromáticas.')
