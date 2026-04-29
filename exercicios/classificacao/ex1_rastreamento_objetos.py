"""
Exercício 1 — Rastreamento de Objetos em Vídeo (com Centroide)
Desenvolva um algoritmo que processe o arquivo de vídeo objetos-coloridos.mov
e realize o rastreamento em tempo real de cada objeto que cruza a tela.
Imprima na imagem o centroide do objeto. Ajuste a velocidade do video para
que possa ser lidas as mensagens.
"""

import cv2
import numpy as np

PATH_VIDEO = '../../videos/objetos-coloridos.mov'

# ── Faixas HSV das cores dos objetos ─────────────────────────────────────────
# Cada cor é definida por uma ou mais faixas [H_min, S_min, V_min] a [H_max, S_max, V_max]
# S_min e V_min altos: excluem pixels acinzentados (dessaturados) e muito escuros (sombras)
# Vermelho: H cruza 0°, por isso usa 2 faixas (0-10 e 160-180)
CORES = {
    'Vermelho': {
        'faixas': [
            (np.array([0,   120, 70]),  np.array([10,  255, 255])),   # H ≈ 0°
            (np.array([160, 120, 70]),  np.array([180, 255, 255])),   # H ≈ 180°
        ],
        'bgr': (0, 0, 220),      # cor BGR para desenho na imagem
    },
    'Verde': {
        'faixas': [(np.array([40, 60, 60]), np.array([85, 255, 255]))],  # H ≈ 60-85°
        'bgr': (0, 200, 0),
    },
    'Azul': {
        'faixas': [(np.array([100, 80, 50]), np.array([130, 255, 255]))],  # H ≈ 120°
        'bgr': (220, 0, 0),
    },
    'Amarelo': {
        'faixas': [(np.array([20, 100, 100]), np.array([35, 255, 255]))],  # H ≈ 30°
        'bgr': (0, 220, 220),
    },
    'Laranja': {
        'faixas': [(np.array([10, 130, 130]), np.array([20, 255, 255]))],  # H ≈ 15-20°
        'bgr': (0, 120, 255),
    },
}

AREA_MIN = 600   # área mínima em px² para considerar objeto real (não ruído)
# Kernel elipse 7×7: bom para objetos circulares; iterations afina a morfologia
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ── Histórico de centroides por cor (para traçar trilha) ──────────────────────
# deque-like: mantém os últimos TRILHA_MAX centroides de cada cor para desenhar rasto
TRILHA_MAX = 30  # quantos pontos manter no histórico
historico = {cor: [] for cor in CORES}

# ── Abertura do vídeo ─────────────────────────────────────────────────────────
# cv2.VideoCapture(caminho): abre arquivo de vídeo
cap = cv2.VideoCapture(PATH_VIDEO)
if not cap.isOpened():
    raise IOError(f'Não foi possível abrir: {PATH_VIDEO}')

# CAP_PROP_FPS: taxa de frames do vídeo original (ex: 30 FPS)
fps_original = cap.get(cv2.CAP_PROP_FPS)
# ── VELOCIDADE REDUZIDA: delay maior = vídeo mais lento ──────────────────────
# delay = ms entre frames; aumentar FATOR_VELOCIDADE acelera, diminuir desacelera
FATOR_VELOCIDADE = 0.3          # 0.3x da velocidade original
delay = max(1, int(1000 / fps_original / FATOR_VELOCIDADE))

print(f'FPS original: {fps_original:.1f} | Reprodução: {fps_original * FATOR_VELOCIDADE:.1f} FPS')
print('Pressione [q] para encerrar | [+]/[-] para ajustar velocidade')

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for hist in historico.values():
            hist.clear()
        continue

    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saida = frame.copy()

    for nome_cor, config in CORES.items():
        # ── Máscara da cor ────────────────────────────────────────────────────
        mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in config['faixas']:
            mascara |= cv2.inRange(hsv, lower, upper)

        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel, iterations=1)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ── Contornos ────────────────────────────────────────────────────────
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            if cv2.contourArea(cnt) < AREA_MIN:
                continue

            # Centroide
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Registrar no histórico (trilha)
            historico[nome_cor].append((cx, cy))
            if len(historico[nome_cor]) > TRILHA_MAX:
                historico[nome_cor].pop(0)

            cor_bgr = config['bgr']

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(saida, (x, y), (x + w, y + h), cor_bgr, 2)

            # Contorno
            cv2.drawContours(saida, [cnt], -1, cor_bgr, 1)

            # Centroide: ponto + texto
            cv2.circle(saida, (cx, cy), 6, cor_bgr, -1)
            cv2.circle(saida, (cx, cy), 8, (255, 255, 255), 1)

            label = f'{nome_cor} ({cx},{cy})'
            cv2.putText(saida, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(saida, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_bgr, 1)

        # ── Desenhar trilha (histórico de posições) ──────────────────────────
        pts = historico[nome_cor]
        for i in range(1, len(pts)):
            alpha = i / len(pts)  # mais brilhante quanto mais recente
            cor_trilha = tuple(int(c * alpha) for c in config['bgr'])
            cv2.line(saida, pts[i - 1], pts[i], cor_trilha, 2)

    # ── HUD ──────────────────────────────────────────────────────────────────
    cv2.putText(saida, f'Velocidade: {FATOR_VELOCIDADE:.1f}x  [+/-]',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(saida, '[q] Sair',
                (10, saida.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow('Rastreamento de Objetos por Cor', saida)

    tecla = cv2.waitKey(delay) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('+'):
        FATOR_VELOCIDADE = min(2.0, FATOR_VELOCIDADE + 0.1)
        delay = max(1, int(1000 / fps_original / FATOR_VELOCIDADE))
        print(f'Velocidade: {FATOR_VELOCIDADE:.1f}x')
    elif tecla == ord('-'):
        FATOR_VELOCIDADE = max(0.1, FATOR_VELOCIDADE - 0.1)
        delay = max(1, int(1000 / fps_original / FATOR_VELOCIDADE))
        print(f'Velocidade: {FATOR_VELOCIDADE:.1f}x')

cap.release()
cv2.destroyAllWindows()

print()
print('=== Técnicas de Rastreamento Utilizadas ===')
print()
print('1. SEGMENTAÇÃO POR COR (HSV + inRange):')
print('   → Cada cor é definida por um intervalo no espaço HSV.')
print('   → Separamos Hue (matiz) de Value (brilho) para resistir a mudanças de luz.')
print()
print('2. CENTROIDE (momentos de imagem):')
print('   → cx = m10/m00  (média ponderada de X pelos valores dos pixels)')
print('   → cy = m01/m00  (média ponderada de Y pelos valores dos pixels)')
print('   → Representa o "centro de massa" do objeto.')
print()
print('3. TRILHA (histórico de centroides):')
print('   → Armazenamos as últimas N posições do centroide.')
print('   → Desenhamos segmentos de reta entre posições consecutivas.')
print('   → Intensidade proporcional à "idade" da posição (mais fraca = mais antiga).')
print()
print('4. AJUSTE DE VELOCIDADE:')
print('   → O delay entre frames controla a velocidade de reprodução.')
print('   → delay = 1000 / fps_original / fator_velocidade')
print('   → Reduzir para 0.3x permite ler as mensagens na tela.')
