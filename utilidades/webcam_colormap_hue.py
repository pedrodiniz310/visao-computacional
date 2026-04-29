"""
Visão Computacional — Webcam com Colormap no Canal Hue
Captura vídeo em tempo real e exibe o canal Hue (matiz)
com o colormap HSV para visualizar a distribuição de cores.
Pressione 'q' para encerrar.
"""

import cv2
import numpy as np

# ── Captura de vídeo ──────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Não foi possível acessar a webcam.')

print('Webcam iniciada. Pressione Q para sair.')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Converter para HSV e extrair canal H ──────────────────────────────────
    # cv2.COLOR_BGR2HSV: converte cada frame BGR → HSV
    # H (matiz): codifica a COR (independente de brilho/saturação)
    # H=0/180=vermelho, H=60=amarelo, H=120=verde, H=180=azul (escala OpenCV 0-179)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Normalizar H de 0-179 para 0-255 para usar com applyColorMap
    # alpha=255/179 ≈ 1.425: fator de escala linear
    h_norm   = cv2.convertScaleAbs(h, alpha=255/179)
    # cv2.applyColorMap(src, colormap): mapeia valores 0-255 para cores falsas
    # COLORMAP_HSV: mapeia o ângulo H para a roda de cores HSV completa
    # Resultado: cada cor real da cena aparece mapeada visualmente para a sua tonalidade
    hue_mapa = cv2.applyColorMap(h_norm, cv2.COLORMAP_HSV)

    # np.hstack: concatena os dois frames horizontalmente numa única janela
    combinado = np.hstack([frame, hue_mapa])
    cv2.imshow('Original | Canal Hue (colormap HSV)  —  Q para sair', combinado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()