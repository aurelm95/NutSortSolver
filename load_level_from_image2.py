import cv2
import numpy as np

# === CONFIGURACIONES ===
ASPECT_RATIO_MIN = 1.2
ASPECT_RATIO_MAX = 3.0
MIN_CONTOUR_HEIGHT=100
ESCALA = 0.5  # Reducción al 50%
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# === FUNCIONES AUXILIARES ===
def redimensionar(imagen, escala=0.5):
    ancho = int(imagen.shape[1] * escala)
    alto = int(imagen.shape[0] * escala)
    return cv2.resize(imagen, (ancho, alto), interpolation=cv2.INTER_AREA)

def nothing(x):
    pass

# === INTERFAZ DE CONTROLES ===
cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controles", 400, 300)

cv2.createTrackbar("H_min", "Controles", 90, 179, nothing)
cv2.createTrackbar("S_min", "Controles", 60, 255, nothing)
cv2.createTrackbar("V_min", "Controles", 50, 255, nothing)
cv2.createTrackbar("H_max", "Controles", 130, 179, nothing)
cv2.createTrackbar("S_max", "Controles", 255, 255, nothing)
cv2.createTrackbar("V_max", "Controles", 155, 255, nothing)

# === CARGAR IMAGEN ===
imagen = cv2.imread("./level31.jpg")
if imagen is None:
    raise FileNotFoundError("No se pudo cargar la imagen.")
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

while True:
    # Leer valores HSV
    h_min = cv2.getTrackbarPos("H_min", "Controles")
    s_min = cv2.getTrackbarPos("S_min", "Controles")
    v_min = cv2.getTrackbarPos("V_min", "Controles")

    h_max = cv2.getTrackbarPos("H_max", "Controles")
    s_max = cv2.getTrackbarPos("S_max", "Controles")
    v_max = cv2.getTrackbarPos("V_max", "Controles")

    lower_blue = np.array([h_min, s_min, v_min])
    upper_blue = np.array([h_max, s_max, v_max])

    # Segmentación del fondo
    mascara_fondo = cv2.inRange(imagen_hsv, lower_blue, upper_blue)
    mascara_objetos = cv2.bitwise_not(mascara_fondo)
    mascara_objetos = cv2.morphologyEx(mascara_objetos, cv2.MORPH_OPEN, KERNEL)
    mascara_objetos = cv2.morphologyEx(mascara_objetos, cv2.MORPH_CLOSE, KERNEL)

    # Contornos y filtrado por aspecto
    # Encontrar contornos válidos
    contornos, _ = cv2.findContours(mascara_objetos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear imagen para mostrar resultados
    resultado = np.zeros_like(imagen)

    indice = 0
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        if w == 0 or h == 0 or h < MIN_CONTOUR_HEIGHT:
            continue
        aspecto = h / w
        if ASPECT_RATIO_MIN <= aspecto <= ASPECT_RATIO_MAX:

            cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), -1)  # dibujar objeto

            # Crear máscara del tamaño del contorno
            mascara_contorno = np.zeros((h, w), dtype=np.uint8)

            # Dibujar el contorno en la máscara, desplazado a (0, 0)
            contorno_desplazado = contorno - [x, y]
            cv2.drawContours(mascara_contorno, [contorno_desplazado], -1, 255, -1)

            # Extraer la región de interés (ROI) de la imagen original
            roi_bgr = imagen[y:y+h, x:x+w]

            # Crear imagen RGBA con fondo transparente
            roi_rgba = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2BGRA)

            # Aplicar la máscara al canal alfa
            roi_rgba[:, :, 3] = mascara_contorno

            # Mostrar o guardar el resultado
            nombre = f"Screw_{indice}.png"
            cv2.imshow(f"Screw {indice}", redimensionar(roi_rgba, ESCALA))
            cv2.imwrite(nombre, roi_rgba)  # Descomenta para guardar
            indice += 1


    # === Mostrar imágenes redimensionadas ===
    cv2.imshow("Original", redimensionar(imagen, ESCALA))
    cv2.imshow("Mascara Fondo Azul", redimensionar(mascara_fondo, ESCALA))
    cv2.imshow("Objetos Filtrados escala y tamaño", redimensionar(resultado, ESCALA))

    # Salir con 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
