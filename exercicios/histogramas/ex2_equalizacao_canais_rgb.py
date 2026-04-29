"""
Exercício 2 — Equalização em 3 Canais (BGR)
Aplique equalização nos 3 canais de uma imagem colorida. Junte os canais e
compare a imagem equalizada com a sem equalização. O que ocorreu?
"""

import cv2
import matplotlib.pyplot as plt

PATH = '../../imagens/frutas.png'

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Equalização em cada canal separadamente ───────────────────────────────────
b, g, r = cv2.split(imagem)

b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)

equalizada = cv2.merge([b_eq, g_eq, r_eq])

# ── Exibição ──────────────────────────────────────────────────────────────────
imagem_rgb   = cv2.cvtColor(imagem,   cv2.COLOR_BGR2RGB)
equalizada_rgb = cv2.cvtColor(equalizada, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Equalização de Histograma nos 3 Canais BGR', fontsize=13)

axes[0].imshow(imagem_rgb)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(equalizada_rgb)
axes[1].set_title('Equalizada (B, G, R separadamente)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# ── Histogramas dos canais ────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 7))
fig2.suptitle('Histogramas por canal — Original vs. Equalizado', fontsize=13)

canais        = [b, g, r]
canais_eq     = [b_eq, g_eq, r_eq]
nomes         = ['Azul (B)', 'Verde (G)', 'Vermelho (R)']
cores         = ['blue', 'green', 'red']

for i in range(3):
    h_orig = cv2.calcHist([canais[i]],    [0], None, [256], [0, 256])
    h_eq   = cv2.calcHist([canais_eq[i]], [0], None, [256], [0, 256])

    axes2[0, i].plot(h_orig, color=cores[i])
    axes2[0, i].set_title(f'{nomes[i]} — Original')
    axes2[0, i].set_xlim([0, 256])

    axes2[1, i].plot(h_eq, color=cores[i])
    axes2[1, i].set_title(f'{nomes[i]} — Equalizado')
    axes2[1, i].set_xlim([0, 256])

plt.tight_layout()
plt.show()

# ── Conclusão ─────────────────────────────────────────────────────────────────
print('=== O que ocorreu? ===')
print()
print('Ao equalizar cada canal R, G e B separadamente, cada canal redistribui')
print('sua intensidade de forma independente. Isso desequilibra a relação entre')
print('os canais, causando uma DISTORÇÃO DE CORES na imagem resultante.')
print()
print('Objetos que eram vermelhos podem ficar azulados, tons de pele ficam')
print('esverdeados, e o resultado final parece "irreal" ou com cores falsas.')
print()
print('Conclusão: equalizar canais RGB separadamente NÃO é uma boa prática para')
print('imagens coloridas, pois altera o balanço cromático da cena.')
