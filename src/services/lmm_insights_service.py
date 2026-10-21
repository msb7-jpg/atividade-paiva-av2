import base64
from io import BytesIO
import json
import logging
from pathlib import Path
from PIL import Image
import httpx

logger = logging.getLogger(__name__)

LAUDO_NAO_GERADO = "[LAUDO NÃO GERADO — LMM indisponível]"


class LMMUnavailableError(RuntimeError):
    """O servidor LMM local não respondeu ou devolveu uma resposta inválida."""


class LMMInsightsService:
    """Consome o modelo multimodal local (OpenAI-compatible) em localhost:8080."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "unsloth/gemma-4-E4B-it-GGUF",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.model = model
        self.timeout = timeout

    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_radiograph_and_heatmap(
        self,
        original_image: Image.Image,
        heatmap_image: Image.Image,
        predicted_class: str,
        confidence: float,
    ) -> str:
        """Envia o raio-X e o mapa de calor Grad-CAM para o LMM local analisar com thinking=False."""
        orig_b64 = self.encode_image_to_base64(original_image)
        heat_b64 = self.encode_image_to_base64(heatmap_image)

        prompt = (
            f"Você é um especialista em radiologia computacional. "
            f"O modelo de rede neural classificou este exame como: {predicted_class} "
            f"com confiança de {confidence * 100:.1f}%.\n\n"
            f"A primeira imagem é a Radiografia de Tórax original. "
            f"A segunda imagem mostra o mapa de calor Grad-CAM sobreposto destacando onde a IA focou.\n\n"
            f"Responda diretamente com 3 tópicos concisos em português:\n"
            f"1. Foco Anatômico: A atenção incide sobre o pulmão ou bordas externas?\n"
            f"2. Coerência com {predicted_class}: O padrão coincide com áreas suspeitas?\n"
            f"3. Conclusão: O diagnóstico do classificador é plausível sob a ótica dos achados visuais?"
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{heat_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
            "chat_template_kwargs": {"thinking": False},
        }

        try:
            client_timeout = httpx.Timeout(self.timeout, connect=3.0)
            with httpx.Client(timeout=client_timeout) as client:
                response = client.post(self.chat_url, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            raise LMMUnavailableError(f"LMM em {self.chat_url} não respondeu: {e!r}") from e

        if not content or not content.strip():
            raise LMMUnavailableError(f"LMM em {self.chat_url} retornou resposta vazia.")
        return content.strip()


def laudo_or_unavailable_notice(
    lmm_service: LMMInsightsService,
    original_image: Image.Image,
    heatmap_image: Image.Image,
    predicted_class: str,
    confidence: float,
) -> str:
    """Obtém o laudo do LMM ou um aviso explícito de que ele não foi gerado (nunca um texto inventado)."""
    try:
        return lmm_service.analyze_radiograph_and_heatmap(
            original_image, heatmap_image, predicted_class=predicted_class, confidence=confidence
        )
    except LMMUnavailableError as e:
        logger.warning(str(e))
        return f"{LAUDO_NAO_GERADO} {e}"
