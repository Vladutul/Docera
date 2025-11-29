import cv2
from PIL import Image

def preprocess_image_crop(crop):
    """
    Applies common image preprocessing steps (grayscale, denoising) to a cropped region.
    """
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
        
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    return denoised

def convert_cv2_to_pil(image_array):
    """Converts a CV2 (numpy array) image to a PIL Image object."""
    # Ensure correct color space if needed, though Image.fromarray usually handles it.
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert BGR to RGB if coming from cv2.imread, though Donut/LayoutLM may expect BGR
        # For simplicity, we assume the preprocessed image is grayscale (1 channel)
        pass 
    return Image.fromarray(image_array)

def normalize_bbox(x, y, bw, bh, w, h):
    """Scales Tesseract bounding box coordinates to the 0-1000 range, as expected by LayoutLMv3."""
    x0 = int(1000 * x / w)
    y0 = int(1000 * y / h)
    x1 = int(1000 * (x + bw) / w)
    y1 = int(1000 * (y + bh) / h)
    return [x0, y0, x1, y1]