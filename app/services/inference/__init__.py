from app.services.inference.errors import InferenceError
from app.services.inference.factory import create_vision_client, create_vision_registry
from app.services.inference.model_ref import format_model_ref, parse_model_ref
from app.services.inference.protocol import VisionLlmClient
from app.services.inference.ollama import OllamaClient, OllamaError
from app.services.inference.openai_compatible import OpenAICompatibleClient, OpenAICompatibleError
from app.services.inference.registry import ResolvedInference, VisionClientRegistry

__all__ = [
    "InferenceError",
    "VisionLlmClient",
    "VisionClientRegistry",
    "ResolvedInference",
    "OllamaClient",
    "OllamaError",
    "OpenAICompatibleClient",
    "OpenAICompatibleError",
    "create_vision_client",
    "create_vision_registry",
    "parse_model_ref",
    "format_model_ref",
]
