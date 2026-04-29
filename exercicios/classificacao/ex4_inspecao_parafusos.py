"""
Exercício 4 — Inspeção de Parafusos (detecção de furos)
O algoritmo deve analisar as imagens parafusos1.png, parafusos2.png e
parafusos3.png. O sistema deve desenhar um quadrado verde com "OK" nas peças
com apenas um furo e um retângulo vermelho com "Defeito" nas peças sem furo
ou com mais de um furo.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGENS = [
    os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/parafusos1.png')),
    os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/parafusos2.png')),
    os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/parafusos3.png')),
]

AREA_MIN_PECA  = 800    # área mínima para considerar uma peça
AREA_MIN_FURO  = 60     # área mínima para considerar um furo real

def processar_parafusos(caminho: str):
    img = cv2.imread(caminho)
    if img is None:
        print(f'  [AVISO] Não encontrada: {caminho}')
        return None, None

    nome = caminho.split('/')[-1]
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Pré-processamento ─────────────────────────────────────────────────────
    # GaussianBlur 5×5: suaviza ruído antes da binarização (reduz falsos furos)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)

    # THRESH_BINARY_INV + THRESH_OTSU: fundo claro → preto, peças escuras → branco
    # THRESH_OTSU: calcula automaticamente o limiar ideal para esta imagem
    limiar, binaria = cv2.threshold(suave, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfologia: limpa ruído e fecha lacunas antes de analisar contornos
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # MORPH_OPEN: remove ruído branco (pontos menores que o kernel)
    limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
    # MORPH_CLOSE: fecha pequenas lacunas dentro da peça
    limpa = cv2.morphologyEx(limpa,   cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # ── Encontrar contornos com hierarquia (RETR_CCOMP) ──────────────────────
    # RETR_CCOMP: 2 níveis de hierarquia:
    #   Nível 0: contornos externos (as peças)
    #   Nível 1: contornos internos (furos dentro das peças)
    # hierarquia[0][i] = [próximo, anterior, primeiro_filho, pai]
    # pai == -1 → contorno externo (peça)
    # pai != -1 → contorno interno (furo)
    contornos, hierarquia = cv2.findContours(limpa, cv2.RETR_CCOMP,
                                             cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:
        print(f'{nome}: nenhum contorno encontrado.')
        return img, img.copy()

    saida    = img.copy()
    relatorio = []

    # ── Processar cada peça (contornos externos, hierarquia[0][i][3] == -1) ──
    for i, cnt in enumerate(contornos):
        # Ignorar furos (contornos com pai != -1)
        if hierarquia[0][i][3] != -1:
            continue

        area_peca = cv2.contourArea(cnt)
        if area_peca < AREA_MIN_PECA:
            continue

        # ── Contar furos DESTE objeto ─────────────────────────────────────────
        n_furos = 0
        j = hierarquia[0][i][2]  # primeiro filho
        while j != -1:
            area_furo = cv2.contourArea(contornos[j])
            if area_furo >= AREA_MIN_FURO:
                n_furos += 1
                # Desenhar contorno do furo em amarelo
                cv2.drawContours(saida, [contornos[j]], -1, (0, 220, 220), 1)
            j = hierarquia[0][j][0]  # próximo irmão

        # ── Bounding box ──────────────────────────────────────────────────────
        x, y, w, h = cv2.boundingRect(cnt)

        # ── Classificação ─────────────────────────────────────────────────────
        if n_furos == 1:
            status     = 'OK'
            cor_rect   = (0, 200, 0)   # verde
            cor_texto  = (0, 200, 0)
            espessura  = 2
        else:
            status     = 'Defeito'
            cor_rect   = (0, 0, 220)   # vermelho
            cor_texto  = (0, 0, 220)
            espessura  = 2

        # Contorno da peça
        cv2.drawContours(saida, [cnt], -1, cor_rect, 2)

        # Quadrado/Retângulo delimitador
        cv2.rectangle(saida, (x - 3, y - 3), (x + w + 3, y + h + 3),
                      cor_rect, espessura)

        # Texto de status
        cv2.putText(saida, status, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(saida, status, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor_texto, 2)

        # Contagem de furos
        cv2.putText(saida, f'furos={n_furos}', (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        relatorio.append({'status': status, 'furos': n_furos, 'area': int(area_peca)})

    # HUD
    ok_count       = sum(1 for r in relatorio if r['status'] == 'OK')
    defeito_count  = sum(1 for r in relatorio if r['status'] == 'Defeito')
    cv2.putText(saida, f'OK: {ok_count}  |  Defeito: {defeito_count}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    print(f'\n{nome}:')
    for idx, r in enumerate(relatorio, 1):
        print(f'  Peça {idx}: {r["status"]:7s} | furos={r["furos"]} | area={r["area"]:,} px²')
    print(f'  → OK: {ok_count}  |  Defeito: {defeito_count}')

    return img, saida

# ── Processar e exibir ────────────────────────────────────────────────────────
print('=== Inspeção de Parafusos ===')

resultados = [(processar_parafusos(c), c.split('/')[-1]) for c in IMAGENS]

fig, axes = plt.subplots(2, len(IMAGENS), figsize=(7 * len(IMAGENS), 12))
fig.suptitle('Inspeção de Parafusos — OK (verde) vs. Defeito (vermelho)', fontsize=13)

for col, ((orig, anot), nome) in enumerate(resultados):
    if orig is None:
        continue
    axes[0, col].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[0, col].set_title(f'Original — {nome}'); axes[0, col].axis('off')

    axes[1, col].imshow(cv2.cvtColor(anot, cv2.COLOR_BGR2RGB))
    axes[1, col].set_title(f'Resultado — {nome}'); axes[1, col].axis('off')

plt.tight_layout()
plt.show()

print()
print('=== Lógica de Inspeção ===')
print()
print('Estrutura de hierarquia (RETR_CCOMP):')
print('  → Nível 0: contornos externos (as peças)')
print('  → Nível 1: contornos internos (os furos dentro das peças)')
print()
print('Critérios de classificação:')
print('  → 0 furos: DEFEITO (parafuso sem furo central)')
print('  → 1 furo:  OK      (parafuso correto)')
print('  → 2+ furos: DEFEITO (parafuso com defeito de fabricação)')
print()
print('Detalhes técnicos:')
print('  → cv2.findContours com RETR_CCOMP retorna a hierarquia em 2 níveis.')
print('  → hierarquia[0][i][2] = índice do primeiro filho (primeiro furo).')
print('  → hierarquia[0][j][0] = próximo irmão do contorno j (próximo furo).')
print('  → Percorremos a lista encadeada de filhos para contar os furos.')
