"""
Visão Computacional — Redução de Resolução e Serrilhado (Aliasing)
Demonstra o efeito de aliasing ao reduzir a resolução espacial
da imagem Chaplin sem anti-aliasing (subamostragem direta).
"""

import os
import cv2
import matplotlib.pyplot as plt

PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../images/chaplin.jpg'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

alt, larg = imagem.shape  # shape de imagem em cinza = (altura, largura) sem canal

# ── Redução de resolução SEM anti-aliasing (serrilhado) ─────────────────────────
def subamostrar(img, fator):
    """Subamostragem direta sem filtro — gera aliasing.

    img[::fator, ::fator]: pega 1 linha a cada 'fator' linhas E
    1 coluna a cada 'fator' colunas → imagem 1/fator² menor.
    Não aplica nenhum filtro antes → viola o teorema de Nyquist →
    gera aliasing (serrilhado) nas bordas diagonais e curvas.
    """
    return img[::fator, ::fator]

r2  = subamostrar(imagem,  2)  # 1/2 da resolução original
r4  = subamostrar(imagem,  4)  # 1/4 da resolução original
r8  = subamostrar(imagem,  8)  # 1/8 da resolução original

# Redimensionar de volta ao tamanho original para comparar visualmente
# cv2.INTER_NEAREST: replica cada pixel sem interpolação → preserva o efeito de "blocos"
# (seria o mesmo que o serrilhado ficasse visível em escala plena)
r2_up = cv2.resize(r2,  (larg, alt), interpolation=cv2.INTER_NEAREST)
r4_up = cv2.resize(r4,  (larg, alt), interpolation=cv2.INTER_NEAREST)
r8_up = cv2.resize(r8,  (larg, alt), interpolation=cv2.INTER_NEAREST)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Redução de Resolução Espacial (Aliasing / Serrilhado)', fontsize=13)

axes[0].imshow(imagem, cmap='gray'); axes[0].set_title(f'Original\n{larg}x{alt}');   axes[0].axis('off')
axes[1].imshow(r2_up,  cmap='gray'); axes[1].set_title(f'1/2 resolução\n{r2.shape[1]}x{r2.shape[0]}'); axes[1].axis('off')
axes[2].imshow(r4_up,  cmap='gray'); axes[2].set_title(f'1/4 resolução\n{r4.shape[1]}x{r4.shape[0]}'); axes[2].axis('off')
axes[3].imshow(r8_up,  cmap='gray'); axes[3].set_title(f'1/8 resolução\n{r8.shape[1]}x{r8.shape[0]}'); axes[3].axis('off')

plt.tight_layout()
plt.show()