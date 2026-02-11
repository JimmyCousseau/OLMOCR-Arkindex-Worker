import base64
import logging
from io import BytesIO

import torch
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from arkindex_worker.models import Element
from arkindex_worker.worker import ElementsWorker

logger = logging.getLogger(__name__)


class OLMoCRModel:
    """Wrapper pour le modèle OLMoCR"""

    def __init__(
        self,
        model_name: str = "allenai/olmOCR-2-7B-1025",
        device: str | None = None,
    ):
        self.device = device or "cpu"
        self.model_name = model_name

        logger.info(f"Chargement du modèle OLMoCR {model_name} sur {self.device}")

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float32,
            )
            .eval()
            .to(self.device)
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        logger.info("OLMoCR prêt à l'emploi (CPU mode)")

    @staticmethod
    def pil_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def predict(self, image: Image.Image, max_tokens: int = 512) -> dict:
        image_base64 = self.pil_to_base64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            temperature=0.1,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            do_sample=True,
        )

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]

        text_output = self.processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )

        return {
            "text": text_output[0].strip(),
            "confidence": None,
        }


class OlmocrWorker(ElementsWorker):
    """
    Worker Arkindex qui applique OLMoCR sur un élément (image)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # On charge le modèle UNE seule fois au lancement du worker
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.olmocr = OLMoCRModel(device=device)

    def _download_element_image(self, element: Element) -> Image.Image:
        """
        Télécharge l'image principale associée à un element Arkindex.
        """

        if not element.files:
            raise ValueError(f"Element {element.id} has no files attached")

        # On prend le premier fichier
        file_obj = element.files[0]

        logger.info(f"Downloading file {file_obj.id} for element {element.id}")

        # Téléchargement via client Arkindex intégré au worker
        response = self.client.request("GET", file_obj.url)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download file {file_obj.id}: {response.status_code} {response.text}"
            )

        image_bytes = response.content
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image

    def process_element(self, element: Element) -> None:
        logger.info(f"Running OLMoCR on element {element.id}")

        image = self._download_element_image(element)

        result = self.olmocr.predict(image=image, max_tokens=512)
        text = result["text"]

        logger.info(f"OCR result for element {element.id}: {text[:200]}...")

        # Créer une annotation Arkindex avec le texte OCR
        self.client.request(
            "POST",
            f"/api/v1/elements/{element.id}/annotations/",
            json={
                "name": "olmocr_transcription",
                "content": text,
                "type": "text",
            },
        )

        logger.info(f"Annotation created for element {element.id}")


def main() -> None:
    OlmocrWorker(description="Worker Arkindex utilisant OLMoCR en CPU-only").run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
