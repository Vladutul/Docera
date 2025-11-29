import pytesseract
import cv2 # Included for type hinting and array manipulation
from ..utils.data_preprocessor import normalize_bbox

class TesseractParser:
    def __init__(self):
        # Tesseract is typically installed globally; no model loading needed here.
        pass

    def parse(self, crop):
        """
        Extracts text tokens and 0-1000 normalized bounding boxes using Tesseract.
        
        Args:
            crop (np.array): The preprocessed image crop (CV2 array).
            
        Returns:
            tuple: (full_text_string, tokens_list, bboxes_list)
        """
        data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT)
        tokens, bboxes = [], []
        h, w = crop.shape[:2]

        for i in range(len(data["text"])):
            word = data["text"][i]
            if not word.strip():
                continue
            
            tokens.append(word)
            
            x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox = normalize_bbox(x, y, bw, bh, w, h)
            bboxes.append(bbox)

        return " ".join(tokens), tokens, bboxes