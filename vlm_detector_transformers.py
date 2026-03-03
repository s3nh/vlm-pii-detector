"""
Document PII Anonymizer using a small Vision-Language Model (Qwen2.5-VL-3B).

Pipeline:
  1. Load scanned document image
  2. Use VLM to detect PII entities with bounding box locations
  3. Apply noise/blur to PII regions
  4. Return anonymized image ready for production LLM processing
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PIIRegion:
    """Represents a detected PII region on the document."""
    label: str          # e.g. "name", "ssn", "email", "phone", "address", "dob"
    bbox: tuple         # (x1, y1, x2, y2) in pixel coordinates
    confidence: float   # 0-1 confidence score


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class VLMPIIDetector:
    """
    Uses Qwen2.5-VL-3B (or similar small VLM) to detect PII
    in scanned document images and return bounding box locations.
    """

    # The prompt instructs the VLM to find PII and return structured output
    DETECTION_PROMPT = """You are a document PII detector. Analyze this scanned document image carefully.

Find ALL personally identifiable information (PII) including:
- Full names of individuals
- Social Security Numbers (SSN) or national ID numbers
- Email addresses
- Phone numbers
- Physical/mailing addresses
- Dates of birth
- Bank account / credit card numbers
- Passport / driver's license numbers
- Signatures

For EACH PII item found, provide:
1. The type of PII (e.g., "name", "ssn", "email", "phone", "address", "dob", "account_number", "signature")
2. The bounding box location as [x1, y1, x2, y2] in relative coordinates (0-1000 scale)
3. The text content

Return your answer as a JSON array. Example:
[
  {"type": "name", "bbox": [100, 50, 400, 80], "text": "John Smith"},
  {"type": "ssn", "bbox": [100, 120, 350, 150], "text": "123-45-6789"}
]

If no PII is found, return an empty array: []
Return ONLY the JSON array, nothing else."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "auto",
    ):
        logger.info(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        self.model.eval()
        logger.info("Model loaded successfully.")

    def detect_pii(self, image: Image.Image) -> list[PIIRegion]:
        """
        Detect PII regions in a document image.

        Args:
            image: PIL Image of the scanned document.

        Returns:
            List of PIIRegion objects with bounding boxes in pixel coordinates.
        """
        w, h = image.size

        # Build the multi-modal message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.DETECTION_PROMPT},
                ],
            }
        ]

        # Prepare inputs
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,      # low temp for deterministic output
                do_sample=False,
            )

        # Decode only the new tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        logger.info(f"VLM raw response:\n{response}")

        return self._parse_response(response, w, h)

    # ---- internal helpers ---------------------------------------------------

    @staticmethod
    def _parse_response(response: str, img_w: int, img_h: int) -> list[PIIRegion]:
        """Parse the JSON response from the VLM into PIIRegion objects."""
        # Try to extract JSON array from the response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON array found in VLM response.")
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse JSON: {exc}")
            return []

        regions: list[PIIRegion] = []
        for item in items:
            bbox_raw = item.get("bbox", [])
            if len(bbox_raw) != 4:
                continue

            # Convert from 0-1000 relative coords → pixel coords
            x1 = int(bbox_raw[0] / 1000 * img_w)
            y1 = int(bbox_raw[1] / 1000 * img_h)
            x2 = int(bbox_raw[2] / 1000 * img_w)
            y2 = int(bbox_raw[3] / 1000 * img_h)

            regions.append(
                PIIRegion(
                    label=item.get("type", "unknown"),
                    bbox=(x1, y1, x2, y2),
                    confidence=item.get("confidence", 0.9),
                )
            )

        logger.info(f"Detected {len(regions)} PII region(s).")
        return regions


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------

class DocumentAnonymizer:
    """
    Applies anonymization (blur / noise / solid fill) over PII regions.
    """

    METHODS = ("blur", "noise", "pixelate", "blackout")

    def __init__(self, method: str = "blur", blur_ksize: int = 51):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self.blur_ksize = blur_ksize   # must be odd

    def anonymize(
        self, image: Image.Image, regions: list[PIIRegion]
    ) -> Image.Image:
        """
        Anonymize PII regions in the image.

        Args:
            image:   Original PIL Image.
            regions: PII regions detected by VLM.

        Returns:
            New PIL Image with PII regions obscured.
        """
        img_array = np.array(image).copy()

        for region in regions:
            x1, y1, x2, y2 = region.bbox
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_array.shape[1], x2)
            y2 = min(img_array.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = img_array[y1:y2, x1:x2]

            if self.method == "blur":
                k = self.blur_ksize
                roi = cv2.GaussianBlur(roi, (k, k), 0)

            elif self.method == "noise":
                noise = np.random.randint(0, 256, roi.shape, dtype=np.uint8)
                roi = cv2.addWeighted(roi, 0.3, noise, 0.7, 0)

            elif self.method == "pixelate":
                small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_LINEAR)
                roi = cv2.resize(
                    small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
                )

            elif self.method == "blackout":
                roi = np.zeros_like(roi)

            img_array[y1:y2, x1:x2] = roi

            logger.info(
                f"  Anonymized [{region.label}] at ({x1},{y1})-({x2},{y2}) "
                f"using {self.method}"
            )

        return Image.fromarray(img_array)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def anonymize_document(
    image_path: str,
    output_path: str | None = None,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    method: str = "blur",
    device: str = "auto",
) -> Image.Image:
    """
    End-to-end: load image → detect PII → anonymize → save/return.

    Args:
        image_path:  Path to scanned document image.
        output_path: Where to save the anonymized image (optional).
        model_name:  HuggingFace model identifier for the VLM.
        method:      Anonymization method: blur | noise | pixelate | blackout.
        device:      Device for inference ("auto", "cpu", "cuda:0").

    Returns:
        Anonymized PIL Image.
    """
    logger.info(f"=== Document PII Anonymization Pipeline ===")
    logger.info(f"Input : {image_path}")
    logger.info(f"Method: {method}")

    # 1. Load image
    image = Image.open(image_path).convert("RGB")
    logger.info(f"Image size: {image.size}")

    # 2. Detect PII
    detector = VLMPIIDetector(model_name=model_name, device=device)
    pii_regions = detector.detect_pii(image)

    if not pii_regions:
        logger.info("No PII detected — returning original image.")
        return image

    # 3. Anonymize
    anonymizer = DocumentAnonymizer(method=method)
    anon_image = anonymizer.anonymize(image, pii_regions)

    # 4. Save
    if output_path is None:
        p = Path(image_path)
        output_path = str(p.with_stem(p.stem + "_anonymized"))

    anon_image.save(output_path)
    logger.info(f"Saved anonymized image to: {output_path}")

    return anon_image


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Anonymize PII in scanned documents using a small VLM."
    )
    parser.add_argument("image", help="Path to the scanned document image.")
    parser.add_argument(
        "-o", "--output", default=None, help="Output path for anonymized image."
    )
    parser.add_argument(
        "-m", "--method",
        choices=DocumentAnonymizer.METHODS,
        default="blur",
        help="Anonymization method (default: blur).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace VLM model name.",
    )
    parser.add_argument(
        "--device", default="auto", help='Device: "auto", "cpu", "cuda:0".'
    )
    args = parser.parse_args()

    anonymize_document(
        image_path=args.image,
        output_path=args.output,
        model_name=args.model,
        method=args.method,
        device=args.device,
    )
