import cv2
import pprint
import numpy as np
import math

imagem = cv2.imread('../imagens/furo1.png')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#Binarização
ret, imgBinarizada = cv2.threshold(imagemCinza, 127, 255, cv2.THRESH_BINARY)

#Contornos
contornos, hierarquia = cv2.findContours(imgBinarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(hierarquia)
objetos = 0
furos = 0

for i in range(len(contornos)):
    if hierarquia[0][i][3] == -1:
        objetos += 1
    else:
        furos += 1

euler = objetos - furos + 1


print(f"Objetos: {objetos}")
print(f"Furos: {furos}")
print(f"Número de Euler: {euler}")