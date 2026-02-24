import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import re
import io
import base64
from typing import List, Tuple, Dict, Optional
import cv2
import matplotlib.pyplot as plt

class PIIAnonymizer:
    def __init__(self, model_path="Qwen/Qwen2-VL-7B-Instruct", device="cuda"):
        """
        Initialize the PII anonymizer with Qwen2-VL model
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...")
        
        # Load model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Define PII patterns to look for
        self.pii_patterns = {
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Simple name pattern
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'date': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            'address': r'\b\d{1,5}\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)\b'
        }
        
    def detect_pii_with_vlm(self, image: Image.Image) -> List[Dict]:
        """
        Use VLM to detect PII regions in the image
        """
        # Prepare prompt for PII detection
        prompt = """Analyze this document image and identify all Personally Identifiable Information (PII). 
        For each PII element found, provide:
        1. The type of PII (name, email, phone, ssn, address, credit card, etc.)
        2. The exact text content
        3. Approximate location in the image (coordinates or region description)
        
        List all PII elements you can find in this document."""
        
        # Prepare messages for chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process the input
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,
                do_sample=False
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the response to extract PII information
        pii_items = self._parse_vlm_response(response)
        
        return pii_items
    
    def _parse_vlm_response(self, response: str) -> List[Dict]:
        """
        Parse VLM response to extract structured PII information
        """
        pii_items = []
        
        # Extract lines that might contain PII information
        lines = response.split('\n')
        for line in lines:
            line = line.lower().strip()
            
            # Check for PII patterns in the response
            for pii_type, pattern in self.pii_patterns.items():
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    for match in matches:
                        pii_items.append({
                            'type': pii_type,
                            'text': match,
                            'confidence': 'high',  # Default confidence
                            'source': 'pattern_match'
                        })
        
        return pii_items
    
    def detect_text_regions(self, image: Image.Image) -> List[Tuple]:
        """
        Use OCR-like approach to detect text regions in the image
        """
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w > 20 and h > 10:
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 10:  # Reasonable aspect ratio for text
                    text_regions.append((x, y, x+w, y+h))
        
        return text_regions
    
    def find_matching_regions(self, image: Image.Image, pii_text: str, 
                             text_regions: List[Tuple]) -> List[Tuple]:
        """
        Find image regions that likely contain the specified PII text
        """
        matching_regions = []
        
        # Convert PIL to OpenCV for OCR attempt
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # For each text region, try to read text using simple template matching
        # This is a simplified approach - in production, you'd want to use Tesseract or similar
        for x1, y1, x2, y2 in text_regions:
            roi = gray[y1:y2, x1:x2]
            
            # Extract potential text from ROI (simplified)
            # In a real implementation, you'd use OCR here
            if roi.size > 0:
                # Simple check for text density
                text_density = np.sum(roi < 128) / roi.size
                if text_density > 0.1:  # Has some text
                    matching_regions.append((x1, y1, x2, y2))
        
        return matching_regions
    
    def blur_regions(self, image: Image.Image, regions: List[Tuple], 
                    blur_radius: int = 10) -> Image.Image:
        """
        Apply Gaussian blur to specified regions in the image
        """
        # Create a copy of the image
        img_copy = image.copy()
        
        # For each region, extract, blur, and paste back
        for x1, y1, x2, y2 in regions:
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            
            if x2 > x1 and y2 > y1:
                # Extract region
                region = img_copy.crop((x1, y1, x2, y2))
                
                # Apply blur
                blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                
                # Paste back
                img_copy.paste(blurred, (x1, y1))
        
        return img_copy
    
    def anonymize_image(self, image_path: str, output_path: str = None, 
                        blur_radius: int = 15) -> Image.Image:
        """
        Main method to anonymize PII in an image
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Detect text regions (as fallback)
        text_regions = self.detect_text_regions(image)
        
        # Use VLM to detect PII
        pii_items = self.detect_pii_with_vlm(image)
        
        # Find regions to blur
        regions_to_blur = []
        
        # Method 1: Use VLM-detected items
        for pii in pii_items:
            if 'text' in pii:
                matching_regions = self.find_matching_regions(image, pii['text'], text_regions)
                regions_to_blur.extend(matching_regions)
        
        # Method 2: If no PII found, use pattern matching on detected text regions
        if not regions_to_blur:
            # Fallback: use regex patterns on the whole image
            # This is a simplified approach
            for region in text_regions:
                regions_to_blur.append(region)
        
        # Apply blur to identified regions
        anonymized_image = self.blur_regions(image, regions_to_blur, blur_radius)
        
        # Save if output path provided
        if output_path:
            anonymized_image.save(output_path)
            print(f"Anonymized image saved to {output_path}")
        
        # Print detection results
        print(f"Detected {len(pii_items)} PII items:")
        for pii in pii_items:
            print(f"  - Type: {pii.get('type', 'unknown')}, Text: {pii.get('text', 'N/A')}")
        
        print(f"Blurred {len(regions_to_blur)} regions")
        
        return anonymized_image

class BatchPIIAnonymizer:
    """Handle batch processing of multiple images"""
    
    def __init__(self, anonymizer: PIIAnonymizer):
        self.anonymizer = anonymizer
    
    def process_folder(self, input_folder: str, output_folder: str, 
                       extensions: List[str] = ['.jpg', '.jpeg', '.png', '.pdf']):
        """
        Process all images in a folder
        """
        import os
        from pathlib import Path
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Process each image
        for file_path in Path(input_folder).iterdir():
            if file_path.suffix.lower() in extensions:
                print(f"\nProcessing: {file_path.name}")
                
                # Generate output path
                output_path = Path(output_folder) / f"anonymized_{file_path.name}"
                
                try:
                    # Anonymize image
                    self.anonymizer.anonymize_image(
                        str(file_path),
                        str(output_path)
                    )
                    print(f"✓ Successfully anonymized: {file_path.name}")
                except Exception as e:
                    print(f"✗ Failed to process {file_path.name}: {str(e)}")

# Usage example
def main():
    # Initialize anonymizer
    anonymizer = PIIAnonymizer(
        model_path="Qwen/Qwen2-VL-7B-Instruct",
        device="cuda"  # or "cpu"
    )
    
    # Single image processing
    input_image = "path/to/your/scanned_document.jpg"
    output_image = "path/to/output/anonymized_document.jpg"
    
    anonymized = anonymizer.anonymize_image(
        input_image, 
        output_image,
        blur_radius=15  # Adjust blur intensity
    )
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(input_image))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(anonymized)
    plt.title('Anonymized Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Batch processing
    # batch_processor = BatchPIIAnonymizer(anonymizer)
    # batch_processor.process_folder(
    #     "input_folder/",
    #     "output_folder/"
    # )

if __name__ == "__main__":
    main()
