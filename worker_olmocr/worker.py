import base64
from io import BytesIO

import torch
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from arkindex_worker import logger
from arkindex_worker.models import Element
from arkindex_worker.worker import ElementsWorker


class OLMoCRModel:
    """Wrapper pour le modèle OLMoCR"""

    def __init__(
        self,
        model_name: str = "allenai/olmOCR-2-7B-1025",
        processor_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str | None = None,
    ):
        self.device = device or "cpu"
        self.model_name = model_name

        logger.info(f"Chargement du modèle OLMoCR {model_name} sur {self.device}")

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
            )
            .eval()
            .to(self.device)
        )

        self.processor = AutoProcessor.from_pretrained(processor_name)

        logger.info(f"OLMoCR prêt (device={self.device})")

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

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=0.1,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]

        text_output = self.processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )

        return {"text": text_output[0].strip(), "confidence": None}


class OlmocrWorker(ElementsWorker):
    """
    Worker Arkindex qui applique OLMoCR sur un élément Arkindex.
    """

    def configure(self):
        super().configure()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.olmocr = OLMoCRModel(device=device)

    def process_element(self, element: Element) -> None:
        logger.info(f"OLMoCR: traitement element {element.id}")

        # récupérer les lignes enfants
        lines = list(element.get_children(type="line"))

        if lines:
            logger.info(f"{len(lines)} lignes détectées pour {element.id}")

            for line in lines:
                image = line.open_image()

                if image is None:
                    continue

                if image.mode != "RGB":
                    image = image.convert("RGB")

                result = self.olmocr.predict(image, max_tokens=256)
                text = result["text"]

                if not text.strip():
                    continue

                self.create_element_transcriptions(
                    line,
                    sub_element_type=line.type,
                    transcriptions=[
                        {
                            "text": text,
                            "polygon": line.polygon,
                            "confidence": 0.5,
                        }
                    ],
                )

        else:
            logger.info("Aucune ligne détectée, OCR page complète")

            image = element.open_image()

            if image is None:
                return

            if image.mode != "RGB":
                image = image.convert("RGB")

            result = self.olmocr.predict(image, max_tokens=512)
            text = result["text"]

            if not text.strip():
                return

            self.create_element_transcriptions(
                element,
                sub_element_type=element.type,
                transcriptions=[
                    {
                        "text": text,
                        "polygon": element.polygon,
                        "confidence": 0.5,
                    }
                ],
            )


def main() -> None:
    OlmocrWorker(description="Worker Arkindex utilisant OLMoCR").run()


if __name__ == "__main__":
    main()
