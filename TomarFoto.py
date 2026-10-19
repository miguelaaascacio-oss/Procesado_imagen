import cv2
import os
import datetime

# Carpeta donde está este script
carpeta = os.path.dirname(os.path.abspath(__file__))

# Abrir la cámara (0 = cámara por defecto, cambia a 1, 2... si tienes varias)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: no se pudo abrir la cámara.")
    exit()

print("Presiona ESPACIO para tomar la foto, o 'q'/ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: no se pudo leer el frame.")
        break

    cv2.imshow("Camara - ESPACIO para capturar", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESPACIO -> guardar foto
    if key == 32:
        # Nombre con fecha y hora para no sobrescribir
        nombre = datetime.datetime.now().strftime("foto_%Y%m%d_%H%M%S.png")
        ruta = os.path.join(carpeta, nombre)
        cv2.imwrite(ruta, frame)
        print(f"Foto guardada en: {ruta}")

    # ESC o 'q' -> salir
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
