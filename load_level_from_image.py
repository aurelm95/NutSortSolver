import cv2
import numpy as np

# Parámetros configurables
ASPECT_RATIO_MIN = 1.2
ASPECT_RATIO_MAX = 3.0

# Cargar imagen
imagen = cv2.imread("./level31.jpg")
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Rango de azul en HSV (ajustable según fondo)
limite_inferior_azul = np.array([90, 60, 50])
limite_superior_azul = np.array([130, 255, 155])

# muestra los colores anteriores


# Crear máscara para el fondo azul
mascara_fondo = cv2.inRange(imagen_hsv, limite_inferior_azul, limite_superior_azul)

# Invertir máscara para obtener objetos
mascara_objetos = cv2.bitwise_not(mascara_fondo)

# Opcional: Limpiar ruido con operaciones morfológicas
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mascara_objetos = cv2.morphologyEx(mascara_objetos, cv2.MORPH_OPEN, kernel)
mascara_objetos = cv2.morphologyEx(mascara_objetos, cv2.MORPH_CLOSE, kernel)

# Encontrar contornos de objetos
contornos, _ = cv2.findContours(mascara_objetos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear imagen para mostrar resultados
resultado = np.zeros_like(imagen)

# Filtrar por relación de aspecto
for contorno in contornos:
    x, y, w, h = cv2.boundingRect(contorno)
    if w == 0:  # evitar división por cero
        continue
    aspecto = h / w
    if ASPECT_RATIO_MIN <= aspecto <= ASPECT_RATIO_MAX:
        cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), -1)  # dibujar objeto

# Mostrar resultados
cv2.imshow("Objetos Segmentados", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
