from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="tls-openai-api")
def tls_openai_api() -> type[ModelAPI]:
    from .inspect_provider import TLSOpenAICompatibleAPI

    return TLSOpenAICompatibleAPI
