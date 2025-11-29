from transformers import DonutProcessor, VisionEncoderDecoderModel
from ..utils.data_preprocessor import convert_cv2_to_pil

class DonutParser:
    def __init__(self, model_name="naver-clova-ix/donut-base"):
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def parse(self, crop):
        """
        Parses an image crop using the Donut model.
        
        Args:
            crop (np.array): The preprocessed image crop (CV2 array).
            
        Returns:
            str: The extracted text string.
        """
        pil_img = convert_cv2_to_pil(crop)
        
        inputs = self.processor(pil_img, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]