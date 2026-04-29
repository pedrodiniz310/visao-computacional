"""
Exercício 2 — Identificação de Figuras Geométricas
Desenvolva um algoritmo que identifique o contorno de cada forma e exiba na
tela o nome da figura geométrica detectada.
Teste com: quadrado.png, circulo.png e triangulo.png
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGENS = {
    'quadrado' : '../../imagens/quadrado.png',
    'circulo'  : '../../imagens/circulo.png',
    'triangulo': '../../imagens/triangulo.png',
}

def identificar_forma(contorno: np.ndarray) -> str:
    """Identifica o nome da forma geométrica pelo número de vértices aproximados."""
    perimetro = cv2.arcLength(contorno, True)
    # Epsilon: tolerância da aproximação (2% do perímetro)
    epsilon    = 0.02 * perimetro
    approx     = cv2.approxPolyDP(contorno, epsilon, True)
    n_vertices = len(approx)

    if n_vertices == 3:
        return 'Triangulo'
    elif n_vertices == 4:
        # Diferenciar quadrado de retângulo pela razão de aspecto
        x, y, w, h = cv2.boundingRect(approx)
        razao = w / float(h)
        return 'Quadrado' if 0.9 <= razao <= 1.1 else 'Retangulo'
    elif n_vertices == 5:
        return 'Pentagono'
    elif n_vertices == 6:
        return 'Hexagono'
    else:
        # Usar circularidade para detectar círculos
        area       = cv2.contourArea(contorno)
        circularidade = 4 * np.pi * area / (perimetro * perimetro)
        if circularidade > 0.8:
            return 'Circulo'
        elif n_vertices > 8:
            return 'Circulo'
        else:
            return f'Poligono ({n_vertices} lados)'

def processar_imagem(caminho: str, nome: str):
    img = cv2.imread(caminho)
    if img is None:
        print(f'  [AVISO] Não encontrada: {caminho}')
        return None, None

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pré-processamento
    suave   = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, bin_ = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfologia para limpar
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    limpa   = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Encontrar contornos
    contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    saida = img.copy()
    formas_detectadas = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        forma = identificar_forma(cnt)
        formas_detectadas.append(forma)

        # Aproximação poligonal
        eps   = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        # Desenhar contorno
        cv2.drawContours(saida, [cnt],   -1, (0,   200, 0),  2)
        cv2.drawContours(saida, [approx], -1, (255, 0,   0),  2)

        # Centroide
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(saida, (cx, cy), 5, (0, 0, 255), -1)

            # Nome da forma
            x_t, y_t = cx - 60, cy - 15
            cv2.putText(saida, forma, (x_t, y_t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(saida, forma, (x_t, y_t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Número de vértices
            eps2   = 0.02 * cv2.arcLength(cnt, True)
            approx2 = cv2.approxPolyDP(cnt, eps2, True)
            cv2.putText(saida, f'v={len(approx2)}', (cx - 15, cy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    print(f'{nome:10s} → {formas_detectadas}')
    return img, saida

# ── Processar e exibir todas as imagens ──────────────────────────────────────
resultados = {}
print('=== Detecção de Figuras Geométricas ===')
for nome, caminho in IMAGENS.items():
    original, anotada = processar_imagem(caminho, nome)
    if original is not None:
        resultados[nome] = (original, anotada)

# ── Plotagem ──────────────────────────────────────────────────────────────────
n = len(resultados)
fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
if n == 1:
    axes = np.array([[axes[0]], [axes[1]]])

fig.suptitle('Identificação de Figuras Geométricas\n'
             'Verde=contorno original | Azul=polígono aproximado', fontsize=12)

for i, (nome, (orig, anot)) in enumerate(resultados.items()):
    axes[0, i].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'Original — {nome}'); axes[0, i].axis('off')

    axes[1, i].imshow(cv2.cvtColor(anot, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'Detectado — {nome}'); axes[1, i].axis('off')

plt.tight_layout()
plt.show()

print()
print('=== Como a identificação funciona ===')
print()
print('cv2.approxPolyDP():')
print('  → Reduz o número de pontos do contorno até que todos os pontos')
print('    restantes estejam a no máximo "epsilon" pixels do contorno original.')
print('  → Quanto maior o epsilon, mais simplificado é o polígono.')
print()
print('Regras de identificação:')
print('  → 3 vértices → Triângulo')
print('  → 4 vértices + razão aspect ≈ 1 → Quadrado')
print('  → 4 vértices + razão aspect ≠ 1 → Retângulo')
print('  → 5 vértices → Pentágono')
print('  → 6 vértices → Hexágono')
print('  → Muitos vértices + circularidade > 0.8 → Círculo')
print()
print('Circularidade = 4π × Área / Perímetro²')
print('  → Círculo perfeito: circularidade = 1.0')
print('  → Quadrado: circularidade ≈ 0.785')
