"""
Exercício 4 — CLAHE com diferentes valores de clipLimit
Encontre uma imagem com iluminação mista (áreas muito claras e muito escuras).
Aplique o CLAHE com diferentes valores de clipLimit (1.0, 2.0 e 5.0) e descreva
o impacto visual no ruído da imagem.
"""

import cv2
import matplotlib.pyplot as plt

PATH = '../../imagens/estrada_escura.png'

# ── Carregamento e conversão para cinza ───────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Equalização global (referência) ──────────────────────────────────────────
equalizada_global = cv2.equalizeHist(cinza)

# ── CLAHE com diferentes clipLimits ──────────────────────────────────────────
resultados = {}
clip_limits = [1.0, 2.0, 5.0]

for clip in clip_limits:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    resultados[clip] = clahe.apply(cinza)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('CLAHE — Impacto do clipLimit na imagem com iluminação mista', fontsize=13)

# Linha superior: imagens
axes[0, 0].imshow(cinza,             cmap='gray'); axes[0, 0].set_title('Original (cinza)');         axes[0, 0].axis('off')
axes[0, 1].imshow(equalizada_global, cmap='gray'); axes[0, 1].set_title('Equalização Global');       axes[0, 1].axis('off')
axes[0, 2].imshow(resultados[1.0],   cmap='gray'); axes[0, 2].set_title('CLAHE clipLimit=1.0\n(suave, menos ruído)'); axes[0, 2].axis('off')

# Linha inferior: CLAHE 2.0 e 5.0 + histograma comparativo
axes[1, 0].imshow(resultados[2.0], cmap='gray'); axes[1, 0].set_title('CLAHE clipLimit=2.0\n(equilíbrio ruído/contraste)'); axes[1, 0].axis('off')
axes[1, 1].imshow(resultados[5.0], cmap='gray'); axes[1, 1].set_title('CLAHE clipLimit=5.0\n(mais contraste, mais ruído)'); axes[1, 1].axis('off')

# Histograma comparativo
axes[1, 2].plot(cv2.calcHist([cinza],             [0], None, [256], [0, 256]), label='Original',    color='black')
axes[1, 2].plot(cv2.calcHist([resultados[1.0]],   [0], None, [256], [0, 256]), label='clip=1.0',   color='blue')
axes[1, 2].plot(cv2.calcHist([resultados[2.0]],   [0], None, [256], [0, 256]), label='clip=2.0',   color='green')
axes[1, 2].plot(cv2.calcHist([resultados[5.0]],   [0], None, [256], [0, 256]), label='clip=5.0',   color='red')
axes[1, 2].set_title('Histogramas comparativos')
axes[1, 2].set_xlim([0, 256])
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# ── Análise ───────────────────────────────────────────────────────────────────
print('=== Impacto do clipLimit no CLAHE ===')
print()
print('O CLAHE (Contrast Limited Adaptive Histogram Equalization) divide a imagem')
print('em células (tiles) e equaliza cada uma localmente. O clipLimit controla')
print('o corte de histograma: picos acima do limite são redistribuídos, limitando')
print('a amplificação de ruído.')
print()
print('clipLimit = 1.0:')
print('  → Corte muito baixo: praticamente não amplifica o histograma.')
print('  → Resultado suave, com POUCO contraste adicional e MÍNIMO ruído.')
print('  → Indicado quando a imagem já tem iluminação razoável.')
print()
print('clipLimit = 2.0 (padrão):')
print('  → Equilibrio entre melhoria de contraste e controle de ruído.')
print('  → Detalhes nas sombras ficam mais visíveis sem introduzir muito ruído.')
print('  → Recomendado para a maioria das aplicações.')
print()
print('clipLimit = 5.0:')
print('  → Corte alto: permite grande amplificação do histograma.')
print('  → MUITO mais contraste nas regiões escuras, mas introduz RUÍDO granulado')
print('     visível, especialmente nas áreas uniformes (céu, paredes).')
print('  → Semelhante à equalização global em comportamento agressivo.')
print()
print('Conclusão: quanto maior o clipLimit, maior o contraste e maior o ruído.')
print('Para imagens com iluminação mista, clipLimit entre 2.0 e 3.0 é ideal.')
