import cv2
import numpy as np

# === CONFIGURACIONES ===
ruta_imagen = "./Screw_2.png"
umbral_cambio = 5.0  # Sensibilidad a cambios de color (ajústalo si es necesario)
min_distancia = 10   # Mínima separación vertical entre cambios (para evitar ruido)

# === CARGAR IMAGEN ===
imagen = cv2.imread(ruta_imagen)
if imagen is None:
    raise FileNotFoundError("No se pudo cargar la imagen.")
alto, ancho, _ = imagen.shape

# Convertir a espacio de color perceptual (LAB o HSV)
imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)

# === Calcular diferencia de color entre filas consecutivas ===
diferencias = []
for y in range(1, alto):
    fila_anterior = imagen_lab[y - 1, :, :].astype(np.int16)
    fila_actual = imagen_lab[y, :, :].astype(np.int16)
    dif = np.mean(np.linalg.norm(fila_actual - fila_anterior, axis=1))  # diferencia promedio por fila
    diferencias.append(dif)

# === Detectar los límites de franjas ===
limites = [0]  # siempre comenzamos desde la fila 0
for i in range(1, len(diferencias)):
    if diferencias[i] > umbral_cambio and (i - limites[-1]) > min_distancia:
        limites.append(i)
limites.append(alto)  # agregar el final de la imagen

# === Extraer franjas y color promedio ===
for i in range(len(limites) - 1):
    y_inicio = limites[i]
    y_fin = limites[i + 1]

    franja = imagen[y_inicio:y_fin, :, :]
    color_promedio = franja.mean(axis=(0, 1)).astype(np.uint8)

    print(f"Franja {i+1}: RGB = {color_promedio[::-1]}")

    # Mostrar el color promedio como una imagen
    muestra = np.full((50, 300, 3), color_promedio, dtype=np.uint8)
    cv2.imshow(f"Franja {i+1}", muestra)

cv2.waitKey(0)
cv2.destroyAllWindows()
