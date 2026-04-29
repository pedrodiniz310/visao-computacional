"""
Exercício 3 — Contagem de Tampas Verdes e Vermelhas
Utilizando as imagens tampinhas1.png e tampinhas2.png, desenvolva um algoritmo
que realize a contagem seletiva de objetos. O sistema deve ser capaz de
diferenciar e imprimir separadamente a quantidade de tampas verdes e vermelhas.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGENS = [
    os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/tampinhas1.png')),
    os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/tampinhas2.png')),
]

# ── Faixas HSV ────────────────────────────────────────────────────────────────
# cv2.inRange(hsv, lower, upper): máscara binária com 255 onde H,S,V ∈ [lower, upper]
# S_min=50/100: exclui tons acinzentados/dessaturados (não são tampas coloridas)
# V_min=50/60:  exclui pixels muito escuros (sombras, reflexos pretos)
FAIXAS_VERDE = [
    (np.array([40,  50,  50]), np.array([85, 255, 255])),  # H≈60-85° = verde
]
# Vermelho cruza H=0°: requer 2 faixas (0-10 e 160-180)
FAIXAS_VERMELHO = [
    (np.array([0,   100, 60]), np.array([10,  255, 255])),   # vermelho próximo a H=0
    (np.array([160, 100, 60]), np.array([180, 255, 255])),   # vermelho próximo a H=180
]

# Área mínima em pixels² — ignora contornos muito pequenos (ruído/sombra)
AREA_MIN = 500

# Kernel elipse 9×9: adequado para objetos circulares como tampas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def criar_mascara_limpa(hsv, faixas):
    """Cria máscara binária combinando múltiplas faixas HSV e limpa com morfologia."""
    mascara = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in faixas:
        # OR bit a bit: combina as duas faixas do vermelho numa única máscara
        mascara |= cv2.inRange(hsv, lower, upper)
    # MORPH_OPEN: remove pontos isolados (reflexos, ruído de segmentação)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel, iterations=2)
    # MORPH_CLOSE: fecha buracos no interior das tampas (reflexo central circular)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mascara

def contar_objetos(mascara, area_min):
    """Retorna lista de contornos externos com área >= area_min."""
    # RETR_EXTERNAL: apenas contornos mais externos (não inclui buracos)
    # CHAIN_APPROX_SIMPLE: comprime pontos redundantes nas retas
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    # Filtra contornos muito pequenos (artefatos de segmentação)
    return [cnt for cnt in contornos if cv2.contourArea(cnt) >= area_min]

def processar_imagem(caminho: str):
    img = cv2.imread(caminho)
    if img is None:
        print(f'  [AVISO] Não encontrada: {caminho}')
        return

    nome_arquivo = caminho.split('/')[-1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Máscaras
    mascara_verde    = criar_mascara_limpa(hsv, FAIXAS_VERDE)
    mascara_vermelho = criar_mascara_limpa(hsv, FAIXAS_VERMELHO)

    # Contornos de cada cor
    cnts_verde    = contar_objetos(mascara_verde,    AREA_MIN)
    cnts_vermelho = contar_objetos(mascara_vermelho, AREA_MIN)

    # Desenhar resultados
    saida = img.copy()

    for cnt in cnts_verde:
        cv2.drawContours(saida, [cnt], -1, (0, 220, 0), 3)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(saida, (cx, cy), 5, (0, 200, 0), -1)
            cv2.putText(saida, 'V', (cx - 8, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for cnt in cnts_vermelho:
        cv2.drawContours(saida, [cnt], -1, (0, 0, 220), 3)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(saida, (cx, cy), 5, (0, 0, 200), -1)
            cv2.putText(saida, 'R', (cx - 8, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # HUD
    texto_verde    = f'Verdes:    {len(cnts_verde)}'
    texto_vermelho = f'Vermelhas: {len(cnts_vermelho)}'
    texto_total    = f'Total:     {len(cnts_verde) + len(cnts_vermelho)}'

    for i, (texto, cor) in enumerate([
        (texto_verde,    (0, 200, 0)),
        (texto_vermelho, (0, 0, 220)),
        (texto_total,    (255, 255, 255)),
    ]):
        cv2.putText(saida, texto, (10, 35 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0),    3)
        cv2.putText(saida, texto, (10, 35 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, cor,           2)

    print(f'\n{nome_arquivo}:')
    print(f'  Tampas VERDES:    {len(cnts_verde)}')
    print(f'  Tampas VERMELHAS: {len(cnts_vermelho)}')
    print(f'  TOTAL:            {len(cnts_verde) + len(cnts_vermelho)}')

    # Plotagem
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f'Contagem de Tampas — {nome_arquivo}', fontsize=12)

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original'); axes[0].axis('off')

    axes[1].imshow(mascara_verde, cmap='gray')
    axes[1].set_title(f'Máscara Verde\n({len(cnts_verde)} tampas)'); axes[1].axis('off')

    axes[2].imshow(mascara_vermelho, cmap='gray')
    axes[2].set_title(f'Máscara Vermelha\n({len(cnts_vermelho)} tampas)'); axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(saida, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Resultado anotado\n(V=verde, R=vermelho)'); axes[3].axis('off')

    plt.tight_layout()
    plt.show()

# ── Executar ──────────────────────────────────────────────────────────────────
print('=== Contagem de Tampas por Cor ===')
for caminho in IMAGENS:
    processar_imagem(caminho)

print()
print('=== Metodologia ===')
print()
print('1. ESPAÇO HSV:')
print('   → O canal H (Hue) representa a cor pura sem influência de brilho.')
print('   → Verde: H ≈ 40-85  | Vermelho: H ≈ 0-10 e 160-180')
print()
print('2. MORFOLOGIA:')
print('   → OPEN (iter=2):  elimina reflexos e pequenos fragmentos de cor.')
print('   → CLOSE (iter=3): une partes de uma mesma tampa separadas por reflexo.')
print()
print('3. CONTAGEM:')
print('   → cv2.findContours com RETR_EXTERNAL: apenas bordas externas.')
print('   → Filtro por área (>500 px²): descarta ruído residual.')
print('   → Cada contorno restante = uma tampa.')
