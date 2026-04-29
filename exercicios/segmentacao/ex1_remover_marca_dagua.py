"""
Exercício 1 — Remoção de Marca D'água

Remova a marca d'água da imagem assinatura.png utilizando segmentação de imagens.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminho absoluto da imagem com marca d'água
PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../images/assinatura.png'))

# ── Carregamento ──────────────────────────────────────────────────────────────
imagem = cv2.imread(PATH)
if imagem is None:
    raise FileNotFoundError(f'Imagem não encontrada: {PATH}')

# Converter para escala de cinza: mais fácil analisar intensidades
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# ── Estratégia: a marca d'água é geralmente mais clara que o texto principal ──
# Usamos limiarização adaptativa para isolar o texto escuro e descartar a marca

# Passo 1: remover ruído com Gaussiano 3×3
# Sigma pequeno preserva bordas do texto enquanto elimina micro-ruído
suave = cv2.GaussianBlur(cinza, (3, 3), 0)

# Passo 2: limiarização global com Otsu
# THRESH_BINARY inverte (fundo claro → preto, texto escuro → branco)
# Otsu calcula automaticamente o melhor limiar para separar texto do fundo
_, bin_otsu = cv2.threshold(suave, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Passo 3: limiarização adaptativa
# ADAPTIVE_THRESH_GAUSSIAN_C: calcula um limiar diferente para cada região
# blockSize=21: tamanho da vizinhança para calcular o limiar local
# C=10: constante subtraída da média ponderada (controla sensibilidade)
# Melhor que global em imagens com iluminação não-uniforme (como documentos)
bin_adapt = cv2.adaptiveThreshold(suave, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 10)

# Passo 4: morfologia para limpar resíduos da marca d'água
# Elemento RECT 2×2 pequeno para não destruir o texto
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# MORPH_OPEN = erosão → dilatação: remove pontos brancos isolados (resíduos de marca)
limpa_otsu  = cv2.morphologyEx(bin_otsu,  cv2.MORPH_OPEN, kernel)
limpa_adapt = cv2.morphologyEx(bin_adapt, cv2.MORPH_OPEN, kernel)

# Passo 5: técnica de inpainting
# A marca d'água tem intensidade intermediária (cinza claro, não branco puro)
# Criar máscara da marca: pixels entre 100 e 180 (não muito escuro, não branco)
_, mascara_marca = cv2.threshold(cinza, 180, 255, cv2.THRESH_BINARY_INV)
_, mascara_texto = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY_INV)
# XOR: remove da máscara os pixels do texto (mantém apenas a marca)
mascara_somente_marca = cv2.bitwise_xor(mascara_marca, mascara_texto)

# Dilatar a máscara para cobrir melhor as bordas da marca
kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mascara_dil = cv2.dilate(mascara_somente_marca, kernel_dil, iterations=1)

# cv2.inpaint(): preenche a região da máscara com base nos pixels ao redor
# inpaintRadius=3: raio de busca por pixels válidos ao redor de cada pixel da máscara
# INPAINT_TELEA: algoritmo baseado em Fast Marching Method (mais suave que Navier-Stokes)
sem_marca = cv2.inpaint(imagem, mascara_dil, inpaintRadius=3,
                         flags=cv2.INPAINT_TELEA)

# ── Exibição ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Remoção de Marca D'água — assinatura.png", fontsize=13)

# Linha superior: imagens originais e máscaras
axes[0, 0].imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original'); axes[0, 0].axis('off')

axes[0, 1].imshow(limpa_otsu, cmap='gray')
axes[0, 1].set_title('Otsu + Abertura Morfológica'); axes[0, 1].axis('off')

axes[0, 2].imshow(limpa_adapt, cmap='gray')
axes[0, 2].set_title('Adaptativa + Abertura'); axes[0, 2].axis('off')

# Linha inferior: máscara e inpainting
axes[1, 0].imshow(mascara_somente_marca, cmap='gray')
axes[1, 0].set_title('Máscara da Marca D'água'); axes[1, 0].axis('off')

axes[1, 1].imshow(mascara_dil, cmap='gray')
axes[1, 1].set_title('Máscara Dilatada'); axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(sem_marca, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Inpainting (TELEA)'); axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
