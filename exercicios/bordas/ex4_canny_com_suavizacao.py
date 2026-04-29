"""
Exercício 4 — Pré-suavização antes do Canny
Aplique o filtro de Mediana ou Gaussiano antes de rodar o algoritmo de Canny.
Prove através de capturas de tela como a remoção prévia de ruído altera a
limpeza das bordas detectadas. Utilize as imagens monumento.png e Noise_SP.png.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGENS = {
    'monumento': '../../imagens/monumento.png',
    'Noise_SP' : '../../imagens/Noise_SP.png',
}

CANNY_T1 = 50
CANNY_T2 = 150

def processar(caminho: str, nome: str) -> None:
    img = cv2.imread(caminho)
    if img is None:
        print(f'  [AVISO] Imagem não encontrada: {caminho}')
        return

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Versões suavizadas ────────────────────────────────────────────────────
    gaussiano = cv2.GaussianBlur(cinza, (5, 5), 0)
    mediana   = cv2.medianBlur(cinza, 5)

    # ── Canny em cada versão ──────────────────────────────────────────────────
    canny_sem    = cv2.Canny(cinza,     CANNY_T1, CANNY_T2)
    canny_gauss  = cv2.Canny(gaussiano, CANNY_T1, CANNY_T2)
    canny_median = cv2.Canny(mediana,   CANNY_T1, CANNY_T2)

    # ── Métricas ──────────────────────────────────────────────────────────────
    px_sem    = int(np.count_nonzero(canny_sem))
    px_gauss  = int(np.count_nonzero(canny_gauss))
    px_median = int(np.count_nonzero(canny_median))

    print(f'\n=== {nome} ===')
    print(f'Pixels de borda sem pré-processamento: {px_sem:,}')
    print(f'Pixels de borda com Gaussiano 5x5:     {px_gauss:,}  ({(px_sem-px_gauss)/px_sem*100:.1f}% menos)')
    print(f'Pixels de borda com Mediana 5x5:       {px_median:,}  ({(px_sem-px_median)/px_sem*100:.1f}% menos)')

    # ── Exibição ──────────────────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Pré-suavização + Canny — {nome}  (T1={CANNY_T1}, T2={CANNY_T2})', fontsize=13)

    # Linha 1: imagens suavizadas
    axes[0, 0].imshow(cinza,     cmap='gray'); axes[0, 0].set_title('Cinza original'); axes[0, 0].axis('off')
    axes[0, 1].imshow(gaussiano, cmap='gray'); axes[0, 1].set_title('Gaussiano 5x5');  axes[0, 1].axis('off')
    axes[0, 2].imshow(mediana,   cmap='gray'); axes[0, 2].set_title('Mediana 5x5');    axes[0, 2].axis('off')

    # Linha 2: resultados do Canny
    axes[1, 0].imshow(canny_sem,    cmap='gray')
    axes[1, 0].set_title(f'Canny sem pré-proc.\n{px_sem:,} pixels de borda'); axes[1, 0].axis('off')

    axes[1, 1].imshow(canny_gauss,  cmap='gray')
    axes[1, 1].set_title(f'Canny + Gaussiano\n{px_gauss:,} pixels de borda'); axes[1, 1].axis('off')

    axes[1, 2].imshow(canny_median, cmap='gray')
    axes[1, 2].set_title(f'Canny + Mediana\n{px_median:,} pixels de borda'); axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

# ── Processar ambas as imagens ────────────────────────────────────────────────
for nome, caminho in IMAGENS.items():
    processar(caminho, nome)

# ── Análise geral ─────────────────────────────────────────────────────────────
print()
print('=== Impacto da Pré-suavização no Canny ===')
print()
print('SEM pré-processamento:')
print('  → O Canny aplica internamente uma suavização Gaussiana leve (ksize=3 implícito).')
print('  → Em imagens com ruído sal-e-pimenta ou ruído gaussiano intenso, muitos')
print('    "falsos positivos" de borda aparecem: variações abruptas de 0→255 no ruído')
print('    geram gradientes altíssimos e são detectadas como bordas reais.')
print()
print('COM Filtro Gaussiano 5x5:')
print('  → Reduz ruído gaussiano/eletrônico (fundo "granuloso" de câmeras).')
print('  → Suaviza transições suaves, o que pode borrar levemente bordas finas.')
print('  → O Canny detecta menos ruído de fundo e produz bordas mais limpas.')
print()
print('COM Filtro de Mediana 5x5:')
print('  → Especialmente eficaz contra ruído SAL E PIMENTA (pixels 0 ou 255 isolados).')
print('  → Preserva melhor as bordas nítidas que o Gaussiano (bordas não são borradas).')
print('  → O Canny vê muito menos falsos positivos de ruído.')
print()
print('CONCLUSÃO:')
print('  → Ruído Sal-e-Pimenta → use Mediana antes do Canny.')
print('  → Ruído Gaussiano     → use Gaussiano antes do Canny.')
print('  → Em ambos os casos, a pré-suavização reduz significativamente o número')
print('    de pixels de borda detectados pelo Canny, tornando o resultado mais limpo.')
