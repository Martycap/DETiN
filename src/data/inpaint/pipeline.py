import cv2

def center_crop(image, size = 255):
    """
    Restituisce l'immagine centrata e ritagliata (per evitare distorsioni) e ridimensionata.
    """
    height, width, _ = image.shape
    min_dim = min(height, width)

    top = (height - min_dim) // 2
    left = (width - min_dim) // 2
    cropped = image[top:top+min_dim, left:left+min_dim]

    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return resized