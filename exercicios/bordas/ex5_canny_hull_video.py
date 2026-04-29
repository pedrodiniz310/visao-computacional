"""
Exercício 5 — Canny + Dilatação + ConvexHull em Vídeo
Crie um script que capture o vídeo e exiba o resultado de Canny. O desafio é
usar a morfologia de Dilatação após o Canny para "engrossar", depois utilize
o operador HULL para detectar objetos.
Vídeos: paca.mp4 e leao.mp4
"""

import cv2
import numpy as np

# ── Configurações ─────────────────────────────────────────────────────────────
VIDEOS = [
    '../../videos/paca.mp4',
    '../../videos/leao.mp4',
]

CANNY_T1       = 50
CANNY_T2       = 130
KERNEL_DIL_SZ  = 3    # tamanho do kernel de dilatação
ITERACOES_DIL  = 2    # iterações de dilatação
AREA_MIN       = 500  # área mínima de contorno para filtrar ruído (px²)

# ── Elemento estruturante para dilatação ──────────────────────────────────────
kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (KERNEL_DIL_SZ, KERNEL_DIL_SZ))

def processar_frame(frame: np.ndarray) -> tuple:
    """Retorna (canny, dilatado, saida_com_hull)."""
    cinza   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # GaussianBlur 5×5: remove ruído de vídeo antes do Canny (câmera tem ruído temporal)
    suave   = cv2.GaussianBlur(cinza, (5, 5), 0)

    # 1. Canny: detecta bordas com histerese (T1=50, T2=130 → razão ~1:2.6)
    bordas = cv2.Canny(suave, CANNY_T1, CANNY_T2)

    # 2. Dilatação — "engrossar" as bordas Canny para fechar lacunas nos contornos
    # Sem dilatação, bordas descontínuas geram muitos fragmentos de contorno
    # Após dilatar, contornos se conectam → menos fragmentos → detecção mais robusta
    dilatado = cv2.dilate(bordas, kernel_dil, iterations=ITERACOES_DIL)

    # 3. Encontrar contornos externos na imagem dilatada
    contornos, _ = cv2.findContours(dilatado, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    saida = frame.copy()
    for cnt in contornos:
        if cv2.contourArea(cnt) < AREA_MIN:
            continue  # ignorar ruído pequeno

        # 4. ConvexHull — envoltória convexa do contorno
        # cv2.convexHull(cnt): retorna os vértices do menor polígono convexo
        # que envolve todos os pontos do contorno (sem concavidades)
        hull = cv2.convexHull(cnt)

        # Desenhar contorno original (verde)
        cv2.drawContours(saida, [cnt],  -1, (0, 200, 0),  1)
        # Desenhar hull (azul) — mostra a forma convexa "simplificada"
        cv2.drawContours(saida, [hull], -1, (200, 0, 0),  2)

        # Centroide: calculado pelos momentos geométricos (m10/m00, m01/m00)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(saida, (cx, cy), 4, (0, 0, 255), -1)

        # Bounding box: menor retângulo alinhado com os eixos que envolve o contorno
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 200, 255), 1)

    # Sobreposição do Canny em amarelo
    saida[bordas > 0] = [0, 220, 220]

    # HUD
    n_obj = sum(1 for c in contornos if cv2.contourArea(c) >= AREA_MIN)
    cv2.putText(saida, f'Objetos: {n_obj}',       (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(saida, f'Canny ({CANNY_T1},{CANNY_T2})', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(saida, 'Verde=contorno  Azul=hull  Ciano=Canny', (10, saida.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return bordas, dilatado, saida


def reproduzir_video(caminho: str) -> None:
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        print(f'[AVISO] Não foi possível abrir: {caminho}')
        return

    fps     = cap.get(cv2.CAP_PROP_FPS)
    delay   = max(1, int(1000 / fps))
    titulo  = caminho.split('/')[-1]

    print(f'\nReproduzindo: {titulo}  (FPS={fps:.1f})')
    print('Tecla [q] = encerrar  |  [p] = pausar/continuar')

    pausado = False
    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        bordas, dilatado, saida = processar_frame(frame)

        # Exibir os 3 canais lado a lado (redimensionados)
        h, w = frame.shape[:2]
        escala = 0.45
        res = (int(w * escala), int(h * escala))

        canny_bgr  = cv2.cvtColor(bordas,   cv2.COLOR_GRAY2BGR)
        dilat_bgr  = cv2.cvtColor(dilatado, cv2.COLOR_GRAY2BGR)

        linha = np.hstack([
            cv2.resize(canny_bgr, res),
            cv2.resize(dilat_bgr, res),
            cv2.resize(saida,     res),
        ])

        cv2.imshow(f'Canny + Dilatacao + Hull — {titulo}', linha)
        cv2.setWindowTitle(f'Canny + Dilatacao + Hull — {titulo}',
                           f'[PAUSADO] {titulo}' if pausado else f'Canny+Dil+Hull — {titulo}')

        tecla = cv2.waitKey(1 if pausado else delay) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado

    cap.release()
    cv2.destroyAllWindows()


# ── Executar para cada vídeo ──────────────────────────────────────────────────
for video in VIDEOS:
    reproduzir_video(video)

print('\n=== Resumo das técnicas utilizadas ===')
print()
print('1. CANNY:')
print('   → Detecta bordas finas com dois limiares de histérese.')
print('   → Bordas espúrias são eliminadas; bordas reais são preservadas.')
print()
print('2. DILATAÇÃO:')
print('   → "Engorda" as bordas detectadas pelo Canny.')
print('   → Conecta bordas que estavam fragmentadas por pequenas lacunas.')
print('   → Facilita o findContours ao criar regiões fechadas.')
print()
print('3. CONVEX HULL:')
print('   → Para cada contorno detectado, cv2.convexHull() cria a envoltória mínima.')
print('   → É como esticar um elástico ao redor do contorno.')
print('   → Útil para objetos com formas irregulares ou parcialmente ocluídos.')
print('   → Permite calcular a área de influência real do objeto na cena.')
