import cv2
import numpy as np
import base64
import json

def get_imgs_b64(imagen_cap_base64, imagenes_ref_base64_json):

    image_cap_base64_data = imagen_cap_base64.split(",")[1]
    image_cap_base64_binary = base64.b64decode(image_cap_base64_data)
    img_cap_base64_decod = cv2.imdecode(np.frombuffer(image_cap_base64_binary, np.uint8), cv2.IMREAD_COLOR)

    # Analizar la cadena JSON
    imagenes_ref_base64 = json.loads(imagenes_ref_base64_json)

    # Inicializar vectores para títulos e imágenes
    imagenes_ref_base64_titles = []
    imagenes_ref_base64_data = []
    
    # Iterar sobre los elementos y extraer el título y la imagen en base64
    for item in imagenes_ref_base64:
        imagenes_ref_base64_titles.append(item['title'])
        imagenes_ref_base64_data.append(item['image_base64'])

    imagenes_ref_base64_decod = [cv2.imdecode(np.frombuffer(base64.b64decode(imagen_base64), np.uint8), cv2.IMREAD_GRAYSCALE) for imagen_base64 in imagenes_ref_base64_data]
    
    #imagenes_base64_data = [item['image_base64'] for item in imagenes_base64]

    #imagenes_referencia = [cv2.imdecode(np.frombuffer(base64.b64decode(imagen_base64), np.uint8), cv2.IMREAD_GRAYSCALE) for imagen_base64 in imagenes_base64_data]
    #imagenes_referencia = [decode_base64_image(item['image_base64']) for item in imagenes_base64]
    
    #img_encoded = base64.b64decode(imagen_ref_base64)
    #img = cv2.imdecode(np.frombuffer(img_encoded, np.uint8), cv2.IMREAD_COLOR)

    #similitud = calcular_similitud(img, img)
    
    
    #Calcula la similitud con cada imagen de referencia y encuentra la máxima
    similitudes = [calcular_similitud(imagen_ref, img_cap_base64_decod) for imagen_ref in imagenes_ref_base64_decod]
    max_similitud = max(similitudes)
    if(max_similitud > 50):
        indice_max_similitud = similitudes.index(max_similitud)  # +1 para que coincida con la imagen
        return "vdeo" + imagenes_ref_base64_titles[indice_max_similitud]
    else:
        return None


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
