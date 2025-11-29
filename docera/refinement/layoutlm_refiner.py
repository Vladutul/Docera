import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from ..utils.data_preprocessor import convert_cv2_to_pil

class LayoutLMRefiner:
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

    def refine(self, crop, tokens, bboxes):
        """
        Applies LayoutLMv3 to tag tokens with semantic labels (NER).
        
        Args:
            crop (np.array): The preprocessed image crop (CV2 array).
            tokens (list): List of text tokens.
            bboxes (list): List of normalized 0-1000 bounding boxes.
            
        Returns:
            str: Tokens tagged with their predicted semantic label.
        """
        pil_img = convert_cv2_to_pil(crop)

        # Prepare inputs (Image, OCR tokens, Bounding boxes)
        enc = self.processor(pil_img, tokens, boxes=bboxes, return_tensors="pt", 
                             truncation=True, padding="max_length", max_length=512)
        
        outputs = self.model(**enc)
        preds = torch.argmax(outputs.logits, dim=2)

        refined_tokens = []
        for token, pred_id in zip(tokens, preds[0].tolist()):
            label = self.model.config.id2label.get(pred_id, "O") 
            refined_tokens.append(f"{token}<{label}>")

        return " ".join(refined_tokens)