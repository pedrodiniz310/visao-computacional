"""
Exercício 1 — Sobel nos Eixos X e Y
Aplique o operador de Sobel apenas no eixo X e depois apenas no eixo Y na
imagem teclado.png. Descreva quais linhas do teclado (horizontais ou
verticais) ficam mais evidentes em cada resultado.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/teclado.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Aplicar Sobel ─────────────────────────────────────────────────────────────

# Sobel X (dx=1, dy=0): detecta variações HORIZONTAIS de intensidade → bordas VERTICAIS
# CV_64F: float64 para preservar gradientes negativos (de claro→escuro)
# ksize=3: kernel Sobel 3×3 (padrão; ksize=1 usa diferenças simples sem suavização)
sobel_x64 = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
# convertScaleAbs: aplica |x| e converte para uint8 (satura em 0-255)
sobel_x    = cv2.convertScaleAbs(sobel_x64)

# Sobel Y (dx=0, dy=1): detecta variações VERTICAIS de intensidade → bordas HORIZONTAIS
sobel_y64 = cv2.Sobel(cinza, cv2.CV_64F, 0, 1, ksize=3)
sobel_y    = cv2.convertScaleAbs(sobel_y64)

# Magnitude combinada (aproximação): 0.5*|Gx| + 0.5*|Gy|
# cv2.addWeighted(src1, alpha, src2, beta, gamma): resultado = alpha*src1 + beta*src2 + gamma
# Alternativa mais precisa: sqrt(Gx² + Gy²), mas addWeighted é mais rápido
sobel_mag = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# ── Exibição ──────────────────────────────────────────────────────────────────
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Operador de Sobel — Eixos X e Y no Teclado', fontsize=13)
axes[0, 0].imshow(imagem_rgb)
axes[0, 0].set_title('Imagem Original'); axes[0, 0].axis('off')
axes[0, 1].imshow(sobel_x, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Sobel X (dx=1, dy=0)\nDetecta bordas VERTICAIS\n'
                      '(variação horizontal de intensidade)'); axes[0, 1].axis('off')
axes[1, 0].imshow(sobel_y, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('Sobel Y (dx=0, dy=1)\nDetecta bordas HORIZONTAIS\n'
                      '(variação vertical de intensidade)'); axes[1, 0].axis('off')
axes[1, 1].imshow(sobel_mag, cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('Magnitude combinada\n|Gx| + |Gy| (0.5 cada)'); axes[1, 1].axis('off')
plt.tight_layout()
plt.show()

# ── Análise ───────────────────────────────────────────────────────────────────
print('=== Análise das Bordas do Teclado ===')
print()
print('Sobel X (derivada horizontal — dx=1, dy=0):')
print('  → Calcula a variação de intensidade da ESQUERDA para a DIREITA.')
print('  → Bordas VERTICAIS (laterais das teclas, separações entre colunas)')
print('    ficam mais evidentes, pois cruzam o gradiente horizontal.')
print('  → As fileiras horizontais do teclado ficam menos visíveis.')
print()
print('Sobel Y (derivada vertical — dx=0, dy=1):')
print('  → Calcula a variação de intensidade de CIMA para BAIXO.')
print('  → Bordas HORIZONTAIS (topo e base das teclas, linhas entre fileiras)')
print('    ficam mais evidentes, pois cruzam o gradiente vertical.')
print('  → As colunas verticais do teclado ficam menos visíveis.')
print()
print('Explicação física:')
print('  → Uma borda vertical separa regiões com cores diferentes lado a lado.')
print('    Nesse caso, a variação ocorre no eixo X → Sobel X a detecta.')
print('  → Uma borda horizontal separa regiões com cores diferentes em cima/baixo.')
print('    Nesse caso, a variação ocorre no eixo Y → Sobel Y a detecta.')
print()
print('Por isso, para detectar TODAS as bordas, combinamos Gx e Gy pela magnitude:')
print('  G = sqrt(Gx² + Gy²)  ou  G ≈ |Gx| + |Gy|  (mais eficiente computacionalmente)')
print('  → Assim, tanto as bordas verticais quanto as horizontais ficam evidentes na imagem combinada.')

