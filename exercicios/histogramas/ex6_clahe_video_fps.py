"""
Exercício 6 — CLAHE em Vídeo em Tempo Real (medição de FPS)
Capture o vídeo paca.mp4 e aplique o realce CLAHE em tempo real.
Meça se há uma queda perceptível no FPS ao adicionar esse processamento.
Como resolver esse problema?
"""

import cv2
import time

PATH_VIDEO = '../../videos/paca.mp4'

# ── Configuração do CLAHE ─────────────────────────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ── Abertura do vídeo ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(PATH_VIDEO)
if not cap.isOpened():
    raise IOError(f'Não foi possível abrir o vídeo: {PATH_VIDEO}')

fps_video = cap.get(cv2.CAP_PROP_FPS)
delay_ms  = max(1, int(1000 / fps_video))

print(f'FPS original do vídeo: {fps_video:.1f}')
print('Pressione [q] para encerrar.')
print('Pressione [c] para ativar/desativar o CLAHE e comparar o FPS.')

# ── Variáveis de medição de FPS ───────────────────────────────────────────────
frame_count   = 0
fps_sem_clahe = 0.0
fps_com_clahe = 0.0
clahe_ativo   = True

t_inicio = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        # Reinicia o vídeo ao chegar no fim
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        t_inicio = time.time()
        frame_count = 0
        continue

    t_frame = time.time()

    if clahe_ativo:
        # ── Aplicar CLAHE apenas no canal V (sem distorção de cores) ──────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = clahe.apply(v)
        hsv_eq = cv2.merge([h, s, v_eq])
        saida = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        modo_texto = 'CLAHE: ATIVO'
        cor_texto  = (0, 200, 0)
    else:
        saida = frame.copy()
        modo_texto = 'CLAHE: INATIVO'
        cor_texto  = (0, 0, 200)

    # ── Calcular FPS em tempo real ────────────────────────────────────────────
    frame_count += 1
    elapsed = time.time() - t_inicio
    fps_atual = frame_count / elapsed if elapsed > 0 else 0

    if clahe_ativo:
        fps_com_clahe = fps_atual
    else:
        fps_sem_clahe = fps_atual

    # ── Sobrepor informações na imagem ────────────────────────────────────────
    cv2.putText(saida, modo_texto,           (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_texto,   2)
    cv2.putText(saida, f'FPS: {fps_atual:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(saida, f'FPS video: {fps_video:.0f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(saida, '[q] Sair | [c] Toggle CLAHE', (10, saida.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('CLAHE em Tempo Real - paca.mp4', saida)

    tecla = cv2.waitKey(delay_ms) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('c'):
        clahe_ativo = not clahe_ativo
        frame_count = 0
        t_inicio = time.time()

cap.release()
cv2.destroyAllWindows()

# ── Relatório ─────────────────────────────────────────────────────────────────
print()
print('=== Relatório de FPS ===')
print(f'FPS sem CLAHE: {fps_sem_clahe:.1f}')
print(f'FPS com CLAHE: {fps_com_clahe:.1f}')
if fps_sem_clahe > 0:
    queda = (1 - fps_com_clahe / fps_sem_clahe) * 100
    print(f'Queda de desempenho: {queda:.1f}%')
print()
print('=== Como resolver a queda de FPS? ===')
print()
print('1. PROCESSAMENTO EM THREAD SEPARADA (Producer-Consumer):')
print('   → Ler frames em uma thread e processar em outra.')
print('   → Evita que o CLAHE bloqueie a captura de frames.')
print()
print('2. REDUZIR RESOLUÇÃO DE PROCESSAMENTO:')
print('   → Aplicar CLAHE em uma versão menor do frame (ex: 50%)')
print('   → Redimensionar de volta para exibição.')
print()
print('3. PROCESSAR FRAMES ALTERNADOS (frame skipping):')
print('   → Aplicar CLAHE em 1 de cada 2 frames (ou 1 de cada 3).')
print('   → Reutilizar o resultado no frame intermediário.')
print()
print('4. USAR GPU / OPENCL:')
print('   → cv2.UMat() transfere o processamento para a GPU via OpenCL.')
print('   → Reduz drasticamente o tempo de execução do CLAHE.')
print()
print('5. AJUSTAR O TAMANHO DO TILE (tileGridSize):')
print('   → Tiles maiores = menos cálculos locais = mais rápido.')
print('   → Ex: tileGridSize=(16,16) em vez de (8,8).')
