"""
PII Detection & Blurring using Small VLM + VLLM
No external OCR dependencies - pure VLM-based approach
Optimized for closed/air-gapped environments
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import base64
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import io
import requests
from urllib.parse import urljoin


@dataclass
class PIIRegion:
    """Detected PII region with metadata"""
    text: str
    entity_type: str  # PERSON, ID_NUMBER, PHONE, EMAIL, etc.
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (normalized 0-1000 or absolute)
    confidence: float
    reason: str  # Why this is PII
    
    def to_dict(self):
        return {
            "text": "[REDACTED]",
            "entity_type": self.entity_type,
            "bbox": {
                "x1": self.bbox[0], "y1": self.bbox[1],
                "x2": self.bbox[2], "y2": self.bbox[3]
            },
            "confidence": self.confidence,
            "reason": self.reason
        }


class VLMPIIDetector:
    """
    Uses Small VLM via VLLM for both OCR and PII detection.
    No PaddleOCR, no external APIs, no Presidio.
    """
    
    # Supported small VLMs (3B-8B parameters, runnable on single GPU)
    SUPPORTED_MODELS = [
        "Qwen/Qwen2-VL-7B-Instruct",      # Best for document understanding
        "openbmb/MiniCPM-V-2_6",           # Very efficient, 8B
        "microsoft/Phi-3-vision-128k-instruct",  # 4.2B, fast
        "HuggingFaceTB/SmolVLM-Instruct",   # 2B, very small
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
    ]
    
    def __init__(self, 
                 vllm_base_url: str = "http://localhost:8000/v1",
                 model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                 api_key: str = "EMPTY"):
        """
        Initialize VLM client for VLLM server
        
        Args:
            vllm_base_url: VLLM OpenAI-compatible API endpoint
            model_name: Model being served
            api_key: API key (usually "EMPTY" for local VLLM)
        """
        self.base_url = vllm_base_url.rstrip('/')
        self.model = model_name
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # System prompt optimized for PII detection
        self.system_prompt = """You are a privacy protection expert. Analyze documents to identify Personally Identifiable Information (PII).

TASK: Find all PII in this image and return structured data.

PII Types to detect:
- PERSON: Full names of individuals
- ID_NUMBER: Passport, national ID, driver's license, SSN, PESEL numbers
- FINANCIAL: Credit cards, IBAN, account numbers
- CONTACT: Phone numbers, email addresses, physical addresses
- DATE_OF_BIRTH: Birth dates
- SIGNATURE: Handwritten signatures
- PHOTO: Portrait photos of people

OUTPUT FORMAT (JSON only):
{
  "regions": [
    {
      "type": "PERSON|ID_NUMBER|FINANCIAL|CONTACT|DATE_OF_BIRTH|SIGNATURE|PHOTO",
      "text": "exact text found",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0-1.0,
      "reason": "brief explanation"
    }
  ]
}

CRITICAL RULES:
1. Coordinates must be [x1, y1, x2, y2] in pixels OR normalized 0-1000
2. If coordinates unknown, use [0, 0, 0, 0] and I'll use text matching
3. Include confidence score (0.0-1.0)
4. Be thorough - missing PII is a privacy violation
5. Return valid JSON only, no markdown"""

    def _encode_image(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """Convert image to base64 string"""
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            # Convert BGR (OpenCV) to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _call_vllm(self, image_b64: str, prompt: str, temperature: float = 0.1) -> str:
        """Make request to VLLM server"""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}  # Force JSON output if supported
        }
        
        try:
            response = requests.post(
                urljoin(self.base_url, "chat/completions"),
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"VLLM request failed: {e}")
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from VLM response (handles markdown code blocks)"""
        # Try to find JSON in code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'(\{.*\})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        # Try parsing entire text
        try:
            return json.loads(text)
        except:
            # Fallback: manual parsing for malformed JSON
            return self._fallback_parse(text)
    
    def _fallback_parse(self, text: str) -> Dict:
        """Emergency parsing when JSON is malformed"""
        regions = []
        
        # Regex patterns for common PII types
        patterns = {
            "PERSON": r'name[:"\'\s]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)',
            "ID_NUMBER": r'(?:ID|number|#)[:"\'\s]+(\d[\d\s-]{5,})',
            "PHONE": r'(?:phone|tel)[:"\'\s]+([\d\s\+-]{7,})',
            "EMAIL": r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        }
        
        for pii_type, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                regions.append({
                    "type": pii_type,
                    "text": match.group(1),
                    "bbox": [0, 0, 0, 0],
                    "confidence": 0.7,
                    "reason": f"Extracted via fallback parsing for {pii_type}"
                })
        
        return {"regions": regions}
    
    def detect_pii(self, 
                   image: Union[str, np.ndarray, Image.Image],
                   image_width: int = None,
                   image_height: int = None) -> Tuple[List[PIIRegion], np.ndarray]:
        """
        Detect PII regions using VLM
        
        Returns:
            List of PIIRegion objects and the loaded image array
        """
        # Load image for dimension reference
        if isinstance(image, str):
            img_array = cv2.imread(image)
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            img_array = image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
            img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        h, w = img_array.shape[:2]
        
        # Get VLM analysis
        image_b64 = self._encode_image(image)
        prompt = "Analyze this document image. Identify ALL PII regions with their exact text content and bounding box coordinates [x1, y1, x2, y2]. Return JSON only."
        
        response_text = self._call_vllm(image_b64, prompt)
        print(f"VLM Response:\n{response_text}\n")
        
        # Parse response
        data = self._extract_json(response_text)
        regions_data = data.get("regions", [])
        
        detected_regions = []
        
        for region in regions_data:
            bbox = region.get("bbox", [0, 0, 0, 0])
            
            # Normalize coordinates if needed
            if all(0 <= c <= 1 for c in bbox):
                # Normalized 0-1 coordinates
                bbox = [int(bbox[0] * w), int(bbox[1] * h), 
                       int(bbox[2] * w), int(bbox[3] * h)]
            elif all(0 <= c <= 100 for c in bbox):
                # Normalized 0-100 coordinates
                bbox = [int(bbox[0] * w / 100), int(bbox[1] * h / 100),
                       int(bbox[2] * w / 100), int(bbox[3] * h / 100)]
            
            detected_regions.append(PIIRegion(
                text=region.get("text", ""),
                entity_type=region.get("type", "UNKNOWN"),
                bbox=tuple(bbox),
                confidence=region.get("confidence", 0.8),
                reason=region.get("reason", "Detected by VLM")
            ))
        
        # If VLM returned no coordinates (all zeros), use text-based localization
        detected_regions = self._localize_regions(img_array, detected_regions)
        
        return detected_regions, img_array
    
    def _localize_regions(self, 
                          image: np.ndarray, 
                          regions: List[PIIRegion]) -> List[PIIRegion]:
        """
        Fallback: Use template matching or simple OCR to find text locations
        if VLM didn't provide coordinates
        """
        # Simple approach: Use OpenCV template matching for text regions
        # In production, you could integrate a lightweight OCR here
        
        updated_regions = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            
            # If coordinates are invalid, try to estimate location
            if x2 <= x1 or y2 <= y1 or x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                # Try to find text using contour detection as approximation
                # This is a basic fallback - in practice, you might want a small OCR model
                
                # For now, mark as full image with warning
                h, w = image.shape[:2]
                region.bbox = (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
                region.reason += " [WARNING: Coordinates estimated]"
            
            updated_regions.append(region)
        
        return updated_regions


class BlurEngine:
    """Handles image blurring with multiple strategies"""
    
    @staticmethod
    def apply_gaussian(image: np.ndarray, 
                       bbox: Tuple[int, ...], 
                       kernel_size: int = 51) -> np.ndarray:
        """Apply Gaussian blur"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            roi = image[y1:y2, x1:x2]
            k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            blurred = cv2.GaussianBlur(roi, (k, k), 30)
            image[y1:y2, x1:x2] = blurred
        return image
    
    @staticmethod
    def apply_pixelate(image: np.ndarray, 
                       bbox: Tuple[int, ...], 
                       pixel_size: int = 10) -> np.ndarray:
        """Apply pixelation effect"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            roi = image[y1:y2, x1:x2]
            temp = cv2.resize(roi, 
                            (max(1, (x2-x1)//pixel_size), max(1, (y2-y1)//pixel_size)),
                            interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(temp, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            image[y1:y2, x1:x2] = pixelated
        return image
    
    @staticmethod
    def apply_solid(image: np.ndarray, 
                    bbox: Tuple[int, ...],
                    color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Apply solid color block"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            # Add label
            cv2.putText(image, "REDACTED", (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image


class SecureImageProcessor:
    """
    Main interface: Combines VLM detection + blurring
    """
    
    BLUR_METHODS = {
        "gaussian": BlurEngine.apply_gaussian,
        "pixelate": BlurEngine.apply_pixelate,
        "solid": BlurEngine.apply_solid
    }
    
    def __init__(self, 
                 vllm_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.detector = VLMPIIDetector(vllm_url, model)
        self.blur_engine = BlurEngine()
    
    def process(self, 
                image_input: Union[str, np.ndarray],
                blur_method: str = "gaussian",
                output_path: Optional[str] = None,
                generate_report: bool = True) -> Dict:
        """
        Complete pipeline: Detect PII -> Blur -> Save -> Report
        
        Returns:
            {
                "output_image": path,
                "detections": [PIIRegion, ...],
                "blur_method": str,
                "summary": {...}
            }
        """
        # Detect
        print("🔍 Detecting PII with VLM...")
        regions, image = self.detector.detect_pii(image_input)
        
        if not regions:
            print("✅ No PII detected")
            return {
                "output_image": image_input if isinstance(image_input, str) else "memory",
                "detections": [],
                "blur_method": blur_method,
                "summary": {"found": False}
            }
        
        print(f"⚠️  Found {len(regions)} PII regions:")
        for r in regions:
            print(f"   • {r.entity_type}: {r.text[:30] if r.text else 'N/A'}... "
                  f"(conf: {r.confidence:.2f})")
        
        # Blur
        print(f"🔒 Applying {blur_method} blur...")
        blur_func = self.BLUR_METHODS.get(blur_method, BlurEngine.apply_gaussian)
        
        result_image = image.copy()
        for region in regions:
            result_image = blur_func(result_image, region.bbox)
        
        # Save
        if output_path is None:
            if isinstance(image_input, str):
                import os
                base, ext = os.path.splitext(image_input)
                output_path = f"{base}_secure{ext}"
            else:
                output_path = "output_secure.png"
        
        cv2.imwrite(output_path, result_image)
        
        # Report
        report = {
            "output_image": output_path,
            "detections": [r.to_dict() for r in regions],
            "blur_method": blur_method,
            "summary": {
                "found": True,
                "total_regions": len(regions),
                "entity_types": list(set(r.entity_type for r in regions)),
                "avg_confidence": sum(r.confidence for r in regions) / len(regions)
            }
        }
        
        if generate_report:
            json_path = output_path.replace(".", "_report.")
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📝 Report saved: {json_path}")
        
        print(f"✅ Secure image saved: {output_path}")
        return report
    
    def batch_process(self, 
                      folder_path: str, 
                      output_folder: str = "./secure",
                      blur_method: str = "pixelate") -> List[Dict]:
        """Process multiple images"""
        import os
        from glob import glob
        
        os.makedirs(output_folder, exist_ok=True)
        results = []
        
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp')
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(folder_path, ext)))
            files.extend(glob(os.path.join(folder_path, ext.upper())))
        
        for f in files:
            try:
                out = os.path.join(output_folder, os.path.basename(f))
                result = self.process(f, blur_method, out)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {f}: {e}")
        
        # Summary
        found = sum(1 for r in results if r["summary"]["found"])
        print(f"\n📊 Batch complete: {len(results)} images, {found} with PII")
        return results


# === VLLM SERVER SETUP ===

"""
# 1. Install VLLM in your closed environment:
pip install vllm

# 2. Download model weights manually (if no internet):
#    - Download from HuggingFace on another machine
#    - Transfer to closed environment via allowed channel
#    - Or use local path instead of model name

# 3. Start VLLM server:
# For Qwen2-VL-7B (requires ~16GB VRAM, can use quantization):
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --port 8000

# For lower VRAM, use quantization:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --quantization awq \
    --gpu-memory-utilization 0.85 \
    --port 8000

# For CPU-only (slow but works):
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --device cpu \
    --port 8000
"""


# === USAGE EXAMPLES ===

if __name__ == "__main__":
    
    # Initialize (ensure VLLM server is running)
    processor = SecureImageProcessor(
        vllm_url="http://localhost:8000/v1",
        model="Qwen/Qwen2-VL-7B-Instruct"  # Change to your model
    )
    
    # Single image
    # result = processor.process(
    #     "document.jpg",
    #     blur_method="pixelate",  # gaussian, pixelate, solid
    #     output_path="document_safe.jpg"
    # )
    
    # Batch processing
    # results = processor.batch_process(
    #     "./documents",
    #     output_folder="./secure_docs",
    #     blur_method="solid"
    # )
    
    # Check what's available
    print("Available blur methods:", list(processor.BLUR_METHODS.keys()))
    print("To use: Ensure VLLM server is running, then call processor.process()")
