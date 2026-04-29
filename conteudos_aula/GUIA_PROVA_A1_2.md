# Guia de Estudos — A1/2: Segmentação e Extração de Metadados
**Data da prova: 29/04/2026 | Peso: 3,0 pts**

---

## O QUE SERÁ COBRADO

### Etapa 1 — Limpeza e Segmentação (1,0 pt)
Converter imagem ruidosa em máscara binária limpa usando:
1. **Filtro de ruído** → suaviza antes de binarizar
2. **Binarização** → objeto branco, fundo preto
3. **Morfologia** → remove pontos isolados no fundo e preenche buracos no objeto

### Etapa 2 — Extração de Características (1,0 pt)
- **Dimensionais**: Área e Perímetro
- **Inerciais**: Centroide (cx, cy) e Momentos de Hu
- **Topológicos**: Número de furos/cavidades em cada componente

### Domínio do Código (1,0 pt)
- Organização do script
- Capacidade de explicar o que cada linha faz

---

## PIPELINE COMPLETO (memorize esta ordem!)

```
Imagem → Cinza → Filtro (Mediana/Gaussiano) → Binarização (Otsu) 
       → Morfologia (Abertura → Fechamento) → findContours 
       → Extração de características
```

---

## FILTROS — QUANDO USAR CADA UM

| Filtro | Função | Ruído ideal | Comando OpenCV |
|--------|--------|-------------|----------------|
| Média | Suaviza uniforme | Eletrônico/gaussiano | `cv2.blur(img, (k,k))` |
| Gaussiano | Suaviza com peso | Eletrônico/gaussiano | `cv2.GaussianBlur(img, (k,k), 0)` |
| **Mediana** | **Preserva bordas** | **Sal e pimenta ✔** | `cv2.medianBlur(img, k)` |
| Bilateral | Preserva bordas + suaviza | Qualquer (lento) | `cv2.bilateralFilter(img, d, sC, sS)` |

> **Dica de prova**: Para imagens ruidosas com sal e pimenta → sempre **Mediana**.

---

## BINARIZAÇÃO

```python
# Limiar fixo
_, binaria = cv2.threshold(cinza, 127, 255, cv2.THRESH_BINARY)

# Otsu (automático) — PREFERIDO
_, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptativa (iluminação irregular)
binaria = cv2.adaptiveThreshold(cinza, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
```

> **Dica de prova**: Sempre tente Otsu primeiro. Ele é automático e funciona bem na maioria dos casos.

---

## MORFOLOGIA MATEMÁTICA

### Operações fundamentais

| Operação | Efeito | Código |
|----------|--------|--------|
| Erosão | Encolhe objetos, remove protuberâncias | `cv2.erode(img, kernel)` |
| Dilatação | Expande objetos, preenche buracos | `cv2.dilate(img, kernel)` |
| **Abertura** (erosão→dilatação) | **Remove ruído branco isolado no fundo** | `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)` |
| **Fechamento** (dilatação→erosão) | **Preenche buracos dentro do objeto** | `cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)` |

### Criar elemento estruturante
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# Formas: MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS
```

### Receita para limpar imagem binarizada:
```python
# 1. Remove pontos brancos isolados no FUNDO (ruído)
limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
# 2. Preenche BURACOS dentro do objeto
limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, kernel)
```

---

## ENCONTRAR CONTORNOS

```python
contornos, hierarquia = cv2.findContours(
    binaria,
    cv2.RETR_CCOMP,      # RETR_CCOMP detecta hierarquia (pai/filho = furos!)
    cv2.CHAIN_APPROX_SIMPLE
)
```

### Modos de recuperação:
| Modo | O que retorna |
|------|---------------|
| `RETR_EXTERNAL` | Só contornos externos (sem furos) |
| `RETR_LIST` | Todos os contornos |
| `RETR_CCOMP` | Hierarquia 2 níveis: objetos + furos |
| `RETR_TREE` | Hierarquia completa |

> **Para contar furos → use `RETR_CCOMP` e analise a hierarquia!**

---

## EXTRAÇÃO DE CARACTERÍSTICAS

### Dimensionais
```python
area      = cv2.contourArea(contorno)
perimetro = cv2.arcLength(contorno, True)
```

### Inerciais — Centroide e Momentos de Hu
```python
M  = cv2.moments(contorno)

# Centroide
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

# Momentos de Hu (7 valores, invariantes a rotação/escala/translação)
hu = cv2.HuMoments(M).flatten()
```

### Topológicas — Contar furos
```python
# Com RETR_CCOMP, hierarquia[0][i] = [próximo, anterior, filho, pai]
# Um furo tem pai != -1 (é filho de outro contorno)

furos = 0
for i in range(len(contornos)):
    # Se o contorno tem um pai, é um furo
    if hierarquia[0][i][3] != -1:
        furos += 1
```

---

## MOMENTOS DE HU — O QUE SÃO

Os 7 Momentos de Hu são invariantes à **translação, rotação e escala**.  
Isso significa que o mesmo objeto sempre gera os mesmos valores, independente de posição ou tamanho.

| Momento | O que descreve |
|---------|---------------|
| Hu[0] | Compactação (círculo vs estrela) |
| Hu[1] | Alongamento (quadrado vs retângulo) |
| Hu[2] | Assimetria em relação a eixo oblíquo |
| Hu[3..6] | Detalhes de forma de alta ordem |

---

## OUTROS COMANDOS ÚTEIS

```python
# Bounding box
x, y, w, h = cv2.boundingRect(contorno)
cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

# Convex Hull
hull = cv2.convexHull(contorno)
cv2.drawContours(img, [hull], -1, (255,0,0), 2)

# Aspect ratio (proporção)
aspecto = w / h

# Extensão (preenchimento dentro do bounding box)
extensao = area / (w * h)

# Solidez (preenchimento dentro do hull)
hull_area = cv2.contourArea(hull)
solidez = area / hull_area
```

---

## DICAS FINAIS PARA A PROVA

1. **Sempre converta para cinza antes de filtrar e binarizar**
   ```python
   cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
   ```

2. **Ordem do pipeline importa**: filtro → binarização → morfologia → contornos → extração

3. **Otsu é seu melhor amigo**: `cv2.THRESH_BINARY + cv2.THRESH_OTSU` — escolhe o limiar automaticamente

4. **Para furos use `RETR_CCOMP`** e conte contornos que têm `hierarquia[0][i][3] != -1`

5. **Evite divisão por zero** ao calcular centroide: verifique `if M['m00'] != 0`

6. **Tamanho do kernel de morfologia**: quanto maior o kernel, mais agressiva a operação — comece com 3x3 ou 5x5

7. **Mediana para sal-e-pimenta**: usa `cv2.medianBlur(img, k)` — `k` deve ser ímpar
