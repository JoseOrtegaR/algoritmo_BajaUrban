import cv2
import numpy as np
import base64

def get_imgs_b64(imagenes_base64):

    #imagenes_decodificadas = [cv2.imdecode(np.frombuffer(base64.b64decode(imagen_base64), np.uint8), cv2.IMREAD_GRAYSCALE) for imagen_base64 in imagenes_base64]

    imagenes_referencia = imagenes_base64[0:len(imagenes_base64) - 2] # imagenes_decodificadas[0:len(imagenes_decodificadas) - 2]
    img = imagenes_base64[len(imagenes_base64) - 1]

    # Calcula la similitud con cada imagen de referencia y encuentra la máxima
    similitudes = [calcular_similitud(imagen_ref, img) for imagen_ref in imagenes_referencia]
    max_similitud = max(similitudes)
    indice_max_similitud = similitudes.index(max_similitud)  # +1 para que coincida con la imagen
    return indice_max_similitud

# Función para calcular la similitud entre dos imágenes
def calcular_similitud(img1, img2):
    sift = cv2.SIFT_create()

    keypoints_img1, descriptores_img1 = sift.detectAndCompute(img1, None)
    keypoints_img2, descriptores_img2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptores_img1, descriptores_img2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    similitud = (len(good_matches) / len(keypoints_img1)) * 100 if len(keypoints_img1) > 0 else 0
    return similitud
