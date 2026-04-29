import cv2
import numpy as np

imagem = cv2.imread('./imagens/quadrado.png')
if imagem is None:
    print('Erro: Imagem não encontrada. Verifique o caminho e o nome do arquivo.')
    exit(1)
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
# Binarização
_, imgBinarizada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY)

contornos, _ = cv2.findContours(imgBinarizada,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

if len(contornos) == 0:
    print('Nenhum contorno encontrado na imagem.')
    exit(1)

contorno = contornos[0]

x, y, w, h = cv2.boundingRect(contorno)
cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)

hull = cv2.convexHull(contorno)
cv2.drawContours(imagem, [hull], -1, (255, 0, 0), 2)

# Cálculos
area = int(cv2.contourArea(contorno))
perimetro = int(cv2.arcLength(contorno, True))
area_caixa_delimitadora = w * h
area_envoltoria_convexa = int(cv2.contourArea(hull))

proporcao = w/h
extensao = area/area_caixa_delimitadora
solidez = area/area_envoltoria_convexa


print(f"Área do objeto: {area} pixels")
print(f"Perímetro: {perimetro} pixels")
print(f"Proporção: {proporcao}")
print(f"Extensão: {extensao}")
print(f"Solidez: {solidez}")
print(f"Área da caixa delimitadora: {area_caixa_delimitadora} pixels")
print(f"Área da envoltória convexa: {area_envoltoria_convexa} pixels")

cv2.imshow('Caixa (azul) e Hull (azul)', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
