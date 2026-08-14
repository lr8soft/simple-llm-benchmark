from __future__ import annotations

from typing import Any

import httpx
from inspect_ai.model._openai import OpenAIAsyncHttpxClient
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI


class TLSOpenAICompatibleAPI(OpenAICompatibleAPI):
    """OpenAI-compatible Inspect provider with configurable TLS verification."""

    def __init__(self, *args: Any, tls_verify: bool = True, **kwargs: Any) -> None:
        if not isinstance(tls_verify, bool):
            raise ValueError("tls_verify must be a bool")
        self.tls_verify = tls_verify
        super().__init__(*args, **kwargs)

    def _create_http_client(self) -> OpenAIAsyncHttpxClient:
        kwargs: dict[str, Any] = {"verify": self.tls_verify}
        if self.client_timeout is not None:
            kwargs["timeout"] = httpx.Timeout(timeout=self.client_timeout, connect=5.0)
        return OpenAIAsyncHttpxClient(**kwargs)
