import os
import cv2
import numpy as np

#matriz K de la camara
K = np.array([[595.0, 0.0, -311.0],
              [0.0, 599.0, -237.0],
              [0.0, 0.0, -1.0]])

#Marcador de 10cm              
Marcador = np.array([[ 5.0,  5.0,  0],		#Puntos ordenados obtenidos en la desnormalizacion Puntos_tres.txt
                   [-5.0,  5.0,  0],
                   [-1.47, 2.80, 0],
                   [ 1.47, 2.80, 0],
                   [-5.0, -5.0,  0],
                   [ 5.0, -5.0,  0]])
                   
directorio = (
    "/home/armas/Documents/TesisMiguelArmas/Tesis_corregida/Prueba_MatrizyPuntos/R_guardadas/"
)

nombre = "0"                          
ruta_txt  = os.path.join(directorio, nombre + ".txt")
ruta_img  = os.path.join(directorio, nombre + ".png") 

# Cargar imagen
img = cv2.imread(ruta_img)
img = cv2.resize(img, (640, 480))
IMG_H = img.shape[0]

with open(ruta_txt, "r") as f:
    datos = np.array(f.read().split(), dtype=float).reshape(-1, 25)

for d in datos:
    etiqueta = int(d[0])
    R = d[1:10].reshape(3, 3)
    t = d[10:13]
    leidos = d[13:25].reshape(6, 2)

    producto = R @ R.T

    p = K @ (R @ Marcador.T + t.reshape(3, 1))  # K (R X + t)
    proyectados = (p[:2] / p[2]).T  # (u, v) = (x/f, y/f)

    print("Id", int(d[0]))
    print("R * R.t\n", producto)
    print("Puntos originales\n", leidos)
    print("Puntos calculados\n", proyectados)
    
    for (u, v) in proyectados:
        v_inv = (IMG_H - 1) - v
        cv2.circle(img, (int(u), int(v_inv)), 4, (0, 0, 255), -1)

    u0, v0 = proyectados[0]
    v0_inv = (IMG_H - 1) - v0
    cv2.putText(img, f"E{etiqueta}", (int(u0) + 10, int(v_inv) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
cv2.imwrite(os.path.join(directorio, nombre + "_puntos.png"), img)
print("Imagen guardada")

