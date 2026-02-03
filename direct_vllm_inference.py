"""
Direct VLLM integration - no server required
Better for resource-constrained closed environments
"""

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import torch

class OfflineVLMPIIDetector:
    """
    Uses VLLM directly without API server
    More efficient for single-machine deployment
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2-VL-7B-Instruct"):
        print(f"Loading model {model_path}... This may take a few minutes.")
        
        # Initialize VLLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True
        )
        
        # Load processor for image preprocessing
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=2000,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        print("✅ Model loaded successfully")
    
    def detect_pii(self, image_path: str) -> List[PIIRegion]:
        """Direct inference without HTTP API"""
        from PIL import Image
        
        # Load and process image
        image = Image.open(image_path)
        
        # Build prompt
        prompt = self.processor.apply_chat_template(
            [{
                "role": "system",
                "content": "You are a PII detection expert. Analyze the image and return JSON with all PII regions."
            }, {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Find all PII. Return JSON with bbox coordinates [x1,y1,x2,y2]."}
                ]
            }],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Run inference
        outputs = self.llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            self.sampling_params
        )
        
        # Parse result
        response_text = outputs[0].outputs[0].text
        return self._parse_response(response_text, image.size)
    
    def _parse_response(self, text: str, img_size: Tuple[int, int]) -> List[PIIRegion]:
        """Parse VLM output to structured data"""
        # Same JSON extraction logic as before
        data = self._extract_json(text)
        regions = []
        
        w, h = img_size
        for r in data.get("regions", []):
            bbox = r.get("bbox", [0,0,0,0])
            # Convert normalized to absolute
            if max(bbox) <= 1.0:
                bbox = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]
            
            regions.append(PIIRegion(
                text=r.get("text", ""),
                entity_type=r.get("type", "UNKNOWN"),
                bbox=tuple(map(int, bbox)),
                confidence=r.get("confidence", 0.8),
                reason=r.get("reason", "")
            ))
        return regions
    
    def _extract_json(self, text: str) -> Dict:
        # Same as previous implementation
        pass
