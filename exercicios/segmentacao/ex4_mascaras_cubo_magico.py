"""
Exercício 4 — Máscaras Binárias do Cubo Mágico (vermelho, amarelo, verde)
Utilize a imagem cubo_magico.png para criar máscaras binárias das cores
vermelha, amarela e verde. Após a binarização, aplique técnicas de Morfologia
para remover ruídos da máscara. Por fim, junte as três imagens em uma.
Exiba a imagem original e as quatro imagens binarizadas.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = '../../imagens/cubo_magico.png'

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# ── Converter para HSV ────────────────────────────────────────────────────────
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# ── Definir faixas HSV para cada cor ─────────────────────────────────────────
# O vermelho cruza o "zero" do canal H (0-10 e 160-180)
FAIXAS = {
    'Vermelho': [
        (np.array([0,   120, 70]),  np.array([10,  255, 255])),
        (np.array([160, 120, 70]),  np.array([180, 255, 255])),
    ],
    'Amarelo': [
        (np.array([20, 100, 100]), np.array([35, 255, 255])),
    ],
    'Verde': [
        (np.array([40, 60, 60]),   np.array([85, 255, 255])),
    ],
}

# ── Elemento estruturante para morfologia ────────────────────────────────────
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def criar_mascara_limpa(hsv_img, faixas):
    """Cria máscara binária para um conjunto de faixas HSV e limpa com morfologia."""
    mascara = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for (lower, upper) in faixas:
        mascara |= cv2.inRange(hsv_img, lower, upper)
    # Abertura: remove pontos pequenos
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel, iterations=2)
    # Fechamento: preenche buracos internos
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mascara

mascaras = {cor: criar_mascara_limpa(hsv, faixas)
            for cor, faixas in FAIXAS.items()}

# ── Aplicar máscaras na imagem original ──────────────────────────────────────
segmentadas = {cor: cv2.bitwise_and(imagem, imagem, mask=mascara)
               for cor, mascara in mascaras.items()}

# ── Juntar as três máscaras em uma única imagem colorida ─────────────────────
mascara_combinada = np.zeros_like(imagem)
CORES_BGR = {
    'Vermelho': (0,   0,   255),
    'Amarelo':  (0,   255, 255),
    'Verde':    (0,   255, 0),
}
for cor, mascara in mascaras.items():
    mascara_combinada[mascara > 0] = CORES_BGR[cor]

# ── Exibição: 5 imagens (original + 3 binárias + combinada) ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('Segmentação por Cor — Cubo Mágico (HSV + Morfologia)', fontsize=13)

axes[0, 0].imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original'); axes[0, 0].axis('off')

axes[0, 1].imshow(mascaras['Vermelho'], cmap='gray')
axes[0, 1].set_title('Máscara Vermelha\n(H: 0-10 e 160-180)'); axes[0, 1].axis('off')

axes[0, 2].imshow(mascaras['Amarelo'], cmap='gray')
axes[0, 2].set_title('Máscara Amarela\n(H: 20-35)'); axes[0, 2].axis('off')

axes[1, 0].imshow(mascaras['Verde'], cmap='gray')
axes[1, 0].set_title('Máscara Verde\n(H: 40-85)'); axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(mascara_combinada, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Máscara Combinada\n(Vermelho + Amarelo + Verde)'); axes[1, 1].axis('off')

# Resultado com segmentação aplicada na imagem real
resultado_overlay = imagem.copy()
resultado_overlay[mascaras['Vermelho'] == 0] = (resultado_overlay[mascaras['Vermelho'] == 0] * 0.3).astype(np.uint8)
resultado_aplicado = cv2.addWeighted(imagem, 0.4, mascara_combinada, 0.6, 0)
axes[1, 2].imshow(cv2.cvtColor(resultado_aplicado, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Sobreposição das 3 cores\nsobre a imagem original'); axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ── Exibição individual com OpenCV ────────────────────────────────────────────
cv2.imshow('Original', imagem)
for cor, img in segmentadas.items():
    cv2.imshow(f'Segmentado — {cor}', img)
cv2.imshow('Mascaras Combinadas', mascara_combinada)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('=== Análise das Máscaras Binárias do Cubo Mágico ===')
print()
for cor, mascara in mascaras.items():
    n_pixels = int(np.count_nonzero(mascara))
    pct = n_pixels / mascara.size * 100
    print(f'Cor {cor:10s}: {n_pixels:7,} pixels ({pct:.1f}% da imagem)')
print()
print('Técnicas utilizadas:')
print('  → cv2.inRange():   cria a máscara binária para a faixa HSV.')
print('  → MORPH_OPEN:      remove manchas e pontos isolados (ruído).')
print('  → MORPH_CLOSE:     preenche buracos internos nas peças do cubo.')
print('  → Para o vermelho: necessárias DUAS faixas pois o matiz "envolve"')
print('    o eixo H=0 (tonalidades de 160-180 + 0-10 são todas vermelhas).')
