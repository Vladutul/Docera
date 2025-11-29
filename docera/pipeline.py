import cv2
import os
import re
import torch
from rapidfuzz import process as fuzz_process

# Import all modular components
from .detectors.yolo_detector import YoloDetector
from .ocr.donut_parser import DonutParser
from .ocr.tesseract_parser import TesseractParser
from .refinement.layoutlm_refiner import LayoutLMRefiner
from .utils.data_preprocessor import preprocess_image_crop

# Models for Classification
from transformers import DistilBertForSequenceClassification, AutoTokenizer

class HybridPipeline:
    """
    Orchestrates the Docera document processing workflow.
    """
    def __init__(self, yolo_model_path=None):
        
        # --- 1. Path Setup ---
        if yolo_model_path is None:
            # Placeholder for where the model should be found in a typical Docera setup
            project_root = os.path.dirname(os.path.abspath(__file__))
            yolo_model_path = os.path.abspath(os.path.join(project_root, '..', '..', 'models', 'best.pt'))
            print(f"Using default YOLO path: {yolo_model_path}")

        # --- 2. Load Core Modules ---
        self.detector = YoloDetector(yolo_model_path) 
        self.donut_parser = DonutParser()
        self.tesseract_parser = TesseractParser()
        self.layoutlm_refiner = LayoutLMRefiner()

        # --- 3. Load DistilBERT Classifier ---
        self.clf_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.clf_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
    # ------------------------------------------------------------------
    # --- Classification & Normalization Steps ---
    # ------------------------------------------------------------------

    def normalize(self, text):
        """Cleans text and attempts to map it to a known entity key (Normalization)."""
        text = re.sub(r"\s+", " ", text)
        candidates = ["TOTAL", "DATE", "SHOP", "ITEM", "ADDRESS", "INVOICE_ID"] 
        match = fuzz_process.extractOne(text, candidates)
        # Use a similarity score threshold (e.g., 80) for a confident match
        entity_key = match[0] if match and match[1] >= 80 else None 
        return text, entity_key

    def classify(self, text):
        """Classifies the text using the DistilBERT model (Classification)."""
        inputs = self.clf_tokenizer(text, return_tensors="pt")
        outputs = self.clf_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        # NOTE: pred is 0 or 1 for the sentiment model loaded (positive/negative)
        return pred

    # ------------------------------------------------------------------
    # --- Main Processing Method ---
    # ------------------------------------------------------------------
    
    def process(self, image_path):
        """
        Runs the full document processing pipeline on a given image file.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
            
        # 1. Object Detection
        # Mocking detection result until YoloDetector is fully implemented
        try:
             results = self.detector.detect(image_path)
        except Exception as e:
            print(f"YoloDetector failed: {e}. Cannot proceed.")
            return {}, img # Return empty fields and original image

        fields = {}
        annotated_img = img.copy()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]

                # 2. Preprocessing
                crop_prep = preprocess_image_crop(crop)

                # 3. OCR & Text Recognition (Donut Primary, Tesseract Fallback)
                try:
                    raw_text = self.donut_parser.parse(crop_prep)
                    # Use Tesseract for bboxes, as Donut is often raw text generation
                    _, tokens, bboxes = self.tesseract_parser.parse(crop_prep) 
                except Exception as e:
                    raw_text, tokens, bboxes = self.tesseract_parser.parse(crop_prep)
                    print(f"Donut failed for {label}: {e}. Using Tesseract.")

                if not tokens:
                    continue

                # 4. Information Refinement (LayoutLMv3 NER)
                refined = self.layoutlm_refiner.refine(crop_prep, tokens, bboxes)
                
                # 5. Normalization & Classification
                normalized, entity_key = self.normalize(refined)
                classification = self.classify(normalized)

                # Store Results
                fields[label] = {
                    "raw_text": raw_text,
                    "refined": refined,
                    "normalized": normalized,
                    "entity_match": entity_key,
                    "classification": classification,
                    "human_review": "required"
                }

                # 6. Visual Output
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0,255,0), 2)
                display_label = f"{label} ({entity_key or 'None'})"
                cv2.putText(annotated_img, display_label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        return fields, annotated_img