import cv2
import numpy as np

def obtener_ancho_visible(ruta_imagen: str) -> int:
    """
    Calcula el ancho visible (no transparente) de una imagen con canal alfa.

    Parámetros:
        ruta_imagen (str): Ruta del archivo de imagen PNG (con canal alfa).

    Retorna:
        int: Ancho (en píxeles) de la región visible del objeto (donde alfa > 0).
             Devuelve 0 si la imagen es completamente transparente.
    """
    # Cargar imagen en modo sin pérdida (con canal alfa)
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)
    
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")
    if imagen.shape[2] != 4:
        raise ValueError("La imagen no tiene canal alfa (no es RGBA).")
    
    # cv2.imshow("Original Screw", imagen)
    imagen = imagen[:-100]
    # cv2.imshow("Cropped Screw without base", imagen)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Extraer canal alfa
    alfa = imagen[:, :, 3]

    # Detectar columnas que tienen al menos un píxel con alfa > 0
    columnas_visibles = np.any(alfa > 0, axis=0)
    indices = np.where(columnas_visibles)[0]

    # Si no hay píxeles visibles, el ancho es 0
    if len(indices) == 0:
        return 0

    # Calcular ancho visible
    x_min = indices[0]
    x_max = indices[-1]
    return x_max - x_min + 1

if __name__ == '__main__':
    from glob import glob

    screw_images_paths=glob(r"C:\Users\aurel\Desktop\Aure\Github\NutSortSolver\Screw_*.png")[:1]
    screw_images_paths=glob("./Screw_*.png")#[:1]

    for screw_image_path in screw_images_paths:
        width=obtener_ancho_visible(screw_image_path)
        print(f"{screw_image_path=}, {width=}")