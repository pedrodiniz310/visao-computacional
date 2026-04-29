"""
Exercício 1 — Remoção de Marca D'água
Remova a marca d'água da imagem assinatura.png utilizando segmentação de imagens.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = '../../imagens/assinatura.png'

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Estratégia: a marca d'água é geralmente mais clara que o texto principal
# Usamos limiarização adaptativa para isolar o texto escuro e descartar a marca
# ──────────────────────────────────────────────────────────────────────────────

# Passo 1: remover ruído
suave = cv2.GaussianBlur(cinza, (3, 3), 0)

# Passo 2: limiarização global com Otsu (destaca o conteúdo mais escuro)
_, bin_otsu = cv2.threshold(suave, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Passo 3: limiarização adaptativa (melhor em iluminação não-uniforme)
bin_adapt = cv2.adaptiveThreshold(suave, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 10)

# Passo 4: morfologia para limpar resíduos da marca d'água
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
limpa_otsu  = cv2.morphologyEx(bin_otsu,  cv2.MORPH_OPEN, kernel)
limpa_adapt = cv2.morphologyEx(bin_adapt, cv2.MORPH_OPEN, kernel)

# Passo 5: técnica de inpainting — usa a máscara da marca para reconstruir
# Detectar a marca d'água como pixels acinzentados (não muito escuros nem brancos)
# A marca tende a ter intensidade intermediária
_, mascara_marca = cv2.threshold(cinza, 180, 255, cv2.THRESH_BINARY_INV)
_, mascara_texto = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY_INV)
mascara_somente_marca = cv2.bitwise_xor(mascara_marca, mascara_texto)

# Dilatar a máscara para cobrir melhor a marca
kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mascara_dil = cv2.dilate(mascara_somente_marca, kernel_dil, iterations=1)

# Inpainting: preenche a região da marca com base nos arredores
sem_marca = cv2.inpaint(imagem, mascara_dil, inpaintRadius=3,
                         flags=cv2.INPAINT_TELEA)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Remoção de Marca D\'água — assinatura.png', fontsize=13)

axes[0, 0].imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original'); axes[0, 0].axis('off')

axes[0, 1].imshow(bin_otsu, cmap='gray')
axes[0, 1].set_title('Binarização Otsu\n(texto + parte da marca)'); axes[0, 1].axis('off')

axes[0, 2].imshow(bin_adapt, cmap='gray')
axes[0, 2].set_title('Binarização Adaptativa\n(separa texto local)'); axes[0, 2].axis('off')

axes[1, 0].imshow(mascara_somente_marca, cmap='gray')
axes[1, 0].set_title('Máscara da marca d\'água\n(tons intermediários)'); axes[1, 0].axis('off')

axes[1, 1].imshow(mascara_dil, cmap='gray')
axes[1, 1].set_title('Máscara dilatada\n(cobre melhor a marca)'); axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(sem_marca, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Resultado — Inpainting TELEA\n(marca removida)'); axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ── Exibição interativa ───────────────────────────────────────────────────────
cv2.imshow('Original', imagem)
cv2.imshow('Sem marca (Inpainting)', sem_marca)
cv2.imshow('Binarizacao Adaptativa', bin_adapt)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('=== Estratégia de remoção de marca d\'água ===')
print()
print('1. BINARIZAÇÃO ADAPTATIVA:')
print('   → Analisa o brilho local em blocos de 21x21 pixels.')
print('   → Pixels com intensidade abaixo da média local são considerados texto.')
print('   → Isola o texto principal sem a marca d\'água (que é mais clara).')
print()
print('2. DETECÇÃO DA MÁSCARA DA MARCA:')
print('   → A marca d\'água tem intensidade intermediária (não branco nem preto).')
print('   → Usando dois limiares e XOR, isolamos apenas a faixa da marca.')
print()
print('3. INPAINTING (cv2.inpaint):')
print('   → Método TELEA: preenchimento baseado em propagação de fronteira.')
print('   → Reconstrói os pixels da marca usando os arredores como referência.')
print('   → Mais eficiente que simplesmente clarear a região (que deixa vestígios).')
