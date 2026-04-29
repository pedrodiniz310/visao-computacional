"""
Exercício 5 — Similaridade Geométrica por Momentos de Hu
Implemente um script para validar a semelhança geométrica entre os objetos
contidos nos arquivos disco1.png, disco2.png, disco3.png, quadrado.png,
circulo.png, triangulo.png e puzzle.png.
A detecção de semelhança deve basear-se nos Momentos e Momentos de Hu,
garantindo que o algoritmo reconheça objetos como semelhantes mesmo com
variações de rotação ou escala.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

IMAGENS = {
    'disco1'   : '../../imagens/disco1.png',
    'disco2'   : '../../imagens/disco2.png',
    'disco3'   : '../../imagens/disco3.png',
    'quadrado' : '../../imagens/quadrado.png',
    'circulo'  : '../../imagens/circulo.png',
    'triangulo': '../../imagens/triangulo.png',
    'puzzle'   : '../../imagens/puzzle.png',
}

LIMIAR_SIMILARIDADE = 0.5  # abaixo desse valor = semelhantes

def extrair_hu_moments(imagem_cinza: np.ndarray) -> np.ndarray:
    """Extrai os 7 Momentos de Hu de uma imagem binária."""
    suave = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    _, binaria = cv2.threshold(suave, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Garantir que o objeto seja branco e o fundo preto
    # (se o fundo for branco, inverter)
    if np.mean(binaria) > 127:
        binaria = cv2.bitwise_not(binaria)

    # Morfologia para limpar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN,  kernel, iterations=1)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Encontrar o maior contorno (objeto principal)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return np.zeros(7), binaria

    cnt_maior = max(contornos, key=cv2.contourArea)
    M = cv2.moments(cnt_maior)

    # Momentos de Hu (log para melhor comparação)
    hu = cv2.HuMoments(M).flatten()
    hu_log = np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return hu_log, binaria

def distancia_hu(hu1: np.ndarray, hu2: np.ndarray) -> float:
    """Calcula a distância entre dois vetores de Momentos de Hu."""
    return cv2.matchShapes.__doc__ and float(np.linalg.norm(hu1 - hu2))
    # Alternativa manual:

def similaridade_matchshapes(img1: np.ndarray, img2: np.ndarray) -> float:
    """Usa cv2.matchShapes para comparar dois contornos diretamente."""
    # Extrai contorno principal de cada imagem
    def contorno_principal(gray):
        _, b = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0),
                             0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(b) > 127:
            b = cv2.bitwise_not(b)
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(cnts, key=cv2.contourArea) if cnts else None

    c1 = contorno_principal(img1)
    c2 = contorno_principal(img2)
    if c1 is None or c2 is None:
        return float('inf')
    return cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)

# ── Carregar imagens ──────────────────────────────────────────────────────────
dados = {}
print('=== Carregando imagens e extraindo Momentos de Hu ===\n')

for nome, caminho in IMAGENS.items():
    img = cv2.imread(caminho)
    if img is None:
        print(f'  [AVISO] Não encontrada: {caminho}')
        continue
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hu, binaria = extrair_hu_moments(cinza)
    dados[nome] = {'img': img, 'cinza': cinza, 'binaria': binaria, 'hu': hu}
    print(f'{nome:10s} → Hu[0]={hu[0]:.3f}  Hu[1]={hu[1]:.3f}  Hu[2]={hu[2]:.3f}')

# ── Matriz de similaridade ────────────────────────────────────────────────────
nomes = list(dados.keys())
n = len(nomes)
mat_sim = np.zeros((n, n))

print('\n=== Matriz de Similaridade (cv2.matchShapes — menor = mais similar) ===\n')
print(' ' * 12 + '  '.join(f'{nm[:8]:8s}' for nm in nomes))

for i, n1 in enumerate(nomes):
    linha = f'{n1:10s} |'
    for j, n2 in enumerate(nomes):
        dist = similaridade_matchshapes(dados[n1]['cinza'], dados[n2]['cinza'])
        mat_sim[i, j] = dist
        linha += f'  {dist:6.3f}'
    print(linha)

# ── Pares mais similares ──────────────────────────────────────────────────────
print(f'\n=== Pares mais semelhantes (dist < {LIMIAR_SIMILARIDADE}) ===\n')
pares_similares = []
for i, j in combinations(range(n), 2):
    d = mat_sim[i, j]
    if d < LIMIAR_SIMILARIDADE:
        pares_similares.append((nomes[i], nomes[j], d))

pares_similares.sort(key=lambda x: x[2])
if pares_similares:
    for n1, n2, d in pares_similares:
        print(f'  {n1:10s} ↔ {n2:10s}  →  dist={d:.4f}  [SEMELHANTE]')
else:
    print('  Nenhum par abaixo do limiar. Aumente LIMIAR_SIMILARIDADE.')

# ── Heatmap da matriz de similaridade ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Similaridade Geométrica por Momentos de Hu', fontsize=13)

im = axes[0].imshow(mat_sim, cmap='RdYlGn_r', vmin=0, vmax=1)
axes[0].set_xticks(range(n)); axes[0].set_xticklabels(nomes, rotation=45, ha='right')
axes[0].set_yticks(range(n)); axes[0].set_yticklabels(nomes)
axes[0].set_title('Matriz de Distância\n(verde=similar, vermelho=diferente)')
plt.colorbar(im, ax=axes[0])

# Adicionar valores no heatmap
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, f'{mat_sim[i,j]:.2f}', ha='center', va='center',
                     fontsize=7, color='black')

# Exibir as imagens em grid
total = len(dados)
cols  = min(4, total)
rows  = (total + cols - 1) // cols
axes[1].axis('off')
axes[1].set_title('Imagens processadas')

plt.tight_layout()
plt.show()

# ── Grid de imagens binárias ──────────────────────────────────────────────────
fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
fig2.suptitle('Contornos binarizados usados para cálculo dos Momentos', fontsize=12)

for idx, (nome, d) in enumerate(dados.items()):
    ax = axes2.flat[idx] if rows > 1 else axes2[idx] if total > 1 else axes2
    ax.imshow(d['binaria'], cmap='gray')
    ax.set_title(nome); ax.axis('off')

for idx in range(total, rows * cols):
    ax = axes2.flat[idx] if rows > 1 else axes2[idx] if total > 1 else axes2
    ax.axis('off')

plt.tight_layout()
plt.show()

print('\n=== Explicação dos Momentos de Hu ===')
print()
print('Os Momentos de Hu são 7 valores INVARIANTES calculados a partir dos')
print('momentos centrais normalizados de uma imagem. Eles são invariantes a:')
print('  → TRANSLAÇÃO: a posição do objeto na imagem não importa.')
print('  → ESCALA: o tamanho do objeto não altera os momentos (normalizados).')
print('  → ROTAÇÃO: a orientação do objeto não altera os momentos.')
print('  → REFLEXÃO: o Hu[6] muda de sinal, mas os outros 6 são invariantes.')
print()
print('Uso na prática:')
print('  → cv2.matchShapes(cnt1, cnt2, method, param):')
print('    Compara dois contornos usando seus Momentos de Hu.')
print('    Retorna 0 para formas idênticas, maior para mais diferentes.')
print()
print('  → cv2.CONTOURS_MATCH_I1:')
print('    dist = Σ |1/hu_A_i - 1/hu_B_i|  (método mais estável)')
print()
print('Aplicações:')
print('  → Reconhecimento de peças industriais em diferentes orientações.')
print('  → Comparação de logos e símbolos.')
print('  → Identificação de caracteres manuscritos (OCR).')
