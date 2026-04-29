"""
Exercício 5 — Detectar Objetos Azuis em Vídeo
Implemente um script que utilize o vídeo objetos-coloridos.mov para
identificar apenas os objetos de cor azul.
"""

import cv2
import numpy as np

PATH_VIDEO = '../../videos/objetos-coloridos.mov'

# ── Faixa HSV para azul ───────────────────────────────────────────────────────
# O canal H (matiz) do azul no OpenCV está entre 100-130 (escala 0-179)
# Equivalente a 200-260° na roda de cores (dividido por 2 para caber em uint8)
# S_min=80: exclui pixels pouco saturados (branco/cinza não é azul)
# V_min=50: exclui pixels muito escuros (sombras, preto não é azul)
AZUL_BAIXO = np.array([100, 80, 50])
AZUL_ALTO  = np.array([130, 255, 255])

# ── Elemento estruturante ─────────────────────────────────────────────────────
# Elipse 7×7: adequado para objetos circulares/arredondados
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ── Abertura do vídeo ─────────────────────────────────────────────────────────
# VideoCapture abre arquivo de vídeo; também aceita índice 0 para webcam
cap = cv2.VideoCapture(PATH_VIDEO)
if not cap.isOpened():
    raise IOError(f'Não foi possível abrir: {PATH_VIDEO}')

# CAP_PROP_FPS: FPS original do arquivo para sincronizar a exibição
fps    = cap.get(cv2.CAP_PROP_FPS)
delay  = max(1, int(1000 / fps))    # waitKey delay em ms por frame
AREA_MIN = 800  # área mínima em px² para considerar contorno real (não ruído)

print('Detectando objetos AZUIS no vídeo...')
print('Pressione [q] para encerrar.')

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # ── Conversão para HSV ────────────────────────────────────────────────────
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ── Máscara para objetos azuis ────────────────────────────────────────────
    mascara = cv2.inRange(hsv, AZUL_BAIXO, AZUL_ALTO)

    # ── Morfologia para limpar a máscara ──────────────────────────────────────
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Encontrar contornos ───────────────────────────────────────────────────
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # ── Resultado: escurecer tudo exceto os objetos azuis ────────────────────
    saida = frame.copy()
    # Escurecer fundo
    fundo_escuro = (frame * 0.3).astype(np.uint8)
    saida = np.where(cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR) > 0,
                     frame, fundo_escuro).astype(np.uint8)

    # Desenhar contornos e informações nos objetos azuis
    n_azul = 0
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN:
            continue
        n_azul += 1

        # Contorno e bounding box
        cv2.drawContours(saida, [cnt], -1, (255, 100, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 200, 255), 2)

        # Centroide
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(saida, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(saida, f'A={int(area)}', (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # HUD
    cv2.putText(saida, f'Objetos azuis: {n_azul}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(saida, 'Faixa HSV azul: H[100-130] S[80-255] V[50-255]',
                (10, saida.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Mostrar a máscara em janela separada
    mascara_vis = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    cv2.putText(mascara_vis, 'Mascara azul (pos-morfologia)', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Deteccao de Objetos Azuis', saida)
    cv2.imshow('Mascara Binaria', mascara_vis)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print()
print('=== Técnicas utilizadas ===')
print()
print('1. ESPAÇO HSV:')
print('   → Superior ao BGR para segmentação por cor pois separa matiz (Hue)')
print('     de luminosidade (Value) e saturação (Saturation).')
print('   → A cor azul tem H ≈ 100-130, independente do brilho ou sombra.')
print()
print('2. cv2.inRange():')
print('   → Gera uma máscara binária: 255 onde o pixel está no intervalo,')
print('     0 caso contrário.')
print()
print('3. MORFOLOGIA (OPEN + CLOSE):')
print('   → OPEN: elimina pontos azuis isolados (reflexos, ruído de cor).')
print('   → CLOSE: preenche buracos dentro dos objetos azuis.')
print()
print('4. DESTAQUE DO FUNDO:')
print('   → Pixels fora da máscara são escurecidos (multiplicação por 0.3).')
print('   → Cria um efeito visual de "spotlight" sobre os objetos azuis.')
