"""
Exercício 2 — Identificação de Figuras Geométricas

Desenvolva um algoritmo que identifique o contorno de cada forma e exiba na
tela o nome da figura geométrica detectada.
Teste com: quadrado.png, circulo.png e triangulo.png
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dicionário de imagens de teste com seus nomes
IMAGENS = {
    'quadrado' : os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/quadrado.png')),
    'circulo'  : os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/circulo.png')),
    'triangulo': os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/triangulo.png')),
}

def identificar_forma(contorno: np.ndarray) -> str:
    """Identifica o nome da forma geométrica pelo número de vértices aproximados.

    Usa cv2.approxPolyDP() para reduzir o contorno a um polígono simplificado
    e conta os vértices resultantes. Círculos produzem muitos vértices e alta
    circularidade; quadrados e triângulos produzem 4 e 3 vértices respectivamente.
    """
    # cv2.arcLength(contorno, fechado): comprimento total do contorno
    # True = contorno fechado (o último ponto liga ao primeiro)
    perimetro = cv2.arcLength(contorno, True)

    # Epsilon: tolerância da aproximação = 2% do perímetro
    # Valor menor → mais vértices (aproximação mais fiel ao contorno original)
    # Valor maior → menos vértices (aproximação mais grosseira = formas mais simples)
    epsilon = 0.02 * perimetro

    # cv2.approxPolyDP(): reduz o contorno ao menor polígono com erro ≤ epsilon
    # Algoritmo de Douglas-Peucker para simplificação de polígonos
    approx     = cv2.approxPolyDP(contorno, epsilon, True)
    n_vertices = len(approx)  # número de vértices do polígono aproximado

    if n_vertices == 3:
        return 'Triangulo'
    elif n_vertices == 4:
        # Diferenciar quadrado de retângulo pela razão de aspecto
        # boundingRect retorna (x, y, largura, altura) do menor retângulo alinhado
        x, y, w, h = cv2.boundingRect(approx)
        razao = w / float(h)
        # Razão entre 0.9 e 1.1 = aproximadamente quadrado (±10% de tolerância)
        return 'Quadrado' if 0.9 <= razao <= 1.1 else 'Retangulo'
    elif n_vertices == 5:
        return 'Pentagono'
    elif n_vertices == 6:
        return 'Hexagono'
    else:
        # Para formas com muitos vértices, verificar circularidade
        # Circularidade = 4π × Área / Perímetro² (1.0 = círculo perfeito)
        area          = cv2.contourArea(contorno)
        circularidade = 4 * np.pi * area / (perimetro * perimetro)
        if circularidade > 0.8:
            return 'Circulo'
        elif n_vertices > 8:
            return 'Circulo'  # muitos vértices = forma arredondada
        else:
            return f'Poligono ({n_vertices} lados)'


def processar_imagem(caminho: str, nome: str):
    """Carrega a imagem, detecta formas e retorna imagem anotada."""
    img = cv2.imread(caminho)
    if img is None:
        print(f'  [AVISO] Não encontrada: {caminho}')
        return None, None

    # Converter para cinza para binarização
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pré-processamento: suavização para remover ruído de borda
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)

    # Binarização inversa com Otsu: objeto escuro → branco (melhor para contornos)
    _, bin_ = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfologia: fecha pequenas lacunas no contorno da forma
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    limpa  = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detectar contornos externos (apenas o mais externo de cada objeto)
    contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    saida = img.copy()
    formas_detectadas = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 500:  # ignora contornos muito pequenos (ruído)
            continue

        # Identifica a forma pelo número de vértices
        forma = identificar_forma(cnt)
        formas_detectadas.append(forma)

        # Aproximação poligonal para visualizar o polígono simplificado
        eps   = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        # Desenha: contorno real (verde) e polígono aproximado (azul)
        cv2.drawContours(saida, [cnt],    -1, (0,   200, 0),  2)  # verde
        cv2.drawContours(saida, [approx], -1, (255, 0,   0),  2)  # azul

        # Centroide: calculado pelos momentos de ordem 0 e 1
        # m10/m00 = x médio; m01/m00 = y médio
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(saida, (cx, cy), 5, (0, 0, 255), -1)  # centroide vermelho

            # Nome da forma sobreposto na imagem (contorno preto + texto ciano para legibilidade)
            x_t, y_t = cx - 60, cy - 15
            cv2.putText(saida, forma, (x_t, y_t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)      # sombra preta
            cv2.putText(saida, forma, (x_t, y_t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # texto ciano

    print(f'{nome}: {formas_detectadas}')
    return saida, formas_detectadas


# ── Processar todas as imagens e exibir ──────────────────────────────────────
resultados = {}
for nome, caminho in IMAGENS.items():
    saida, formas = processar_imagem(caminho, nome)
    if saida is not None:
        resultados[nome] = (saida, formas)

# Subplot com todas as imagens processadas
n = len(resultados)
if n > 0:
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle('Identificação de Figuras Geométricas', fontsize=13)

    for ax, (nome, (saida, formas)) in zip(axes, resultados.items()):
        ax.imshow(cv2.cvtColor(saida, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{nome.capitalize()}\nDetectado: {", ".join(formas)}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
