import cv2
import numpy as np
import base64

def decode_base64_image(image_base64):
    try:
        # Agregar caracteres de relleno si es necesario
        while len(image_base64) % 4 != 0:
            image_base64 += '='
        
        decoded_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        return image
    except Exception as e:
        print(f"Error al decodificar la imagen: {str(e)}")
        return None


def get_imgs_b64(imagen_ref_base64, imagenes_base64):

    #imagenes_base64_fila0 = [fila[0] for fila in imagenes_base64]
    #imagenes_base64_fila1 = [fila[1] for fila in imagenes_base64]

    #imagenes_referencia = [cv2.imdecode(np.frombuffer(base64.b64decode(imagen_base64), np.uint8), cv2.IMREAD_GRAYSCALE) for imagen_base64 in imagenes_base64_fila0]
    
    #imagenes_base64_data = [item['image_base64'] for item in imagenes_base64]

    #imagenes_referencia = [cv2.imdecode(np.frombuffer(base64.b64decode(imagen_base64), np.uint8), cv2.IMREAD_GRAYSCALE) for imagen_base64 in imagenes_base64_data]
    #imagenes_referencia = [decode_base64_image(item['image_base64']) for item in imagenes_base64]
    
    img_encoded = base64.b64decode(imagen_ref_base64)
    img = cv2.imdecode(np.frombuffer(img_encoded, np.uint8), cv2.IMREAD_COLOR)

    similitud = calcular_similitud(img, img)
    
    
    # Calcula la similitud con cada imagen de referencia y encuentra la máxima
    #similitudes = [calcular_similitud(imagen_ref, img) for imagen_ref in imagenes_referencia]
    #max_similitud = max(similitudes)
    #if(max_similitud > 50):
    #    indice_max_similitud = similitudes.index(max_similitud)  # +1 para que coincida con la imagen
    #    return imagenes_base64_fila1[indice_max_similitud]
    #else:
    #    return None
    return 1  #imagenes_referencia

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
