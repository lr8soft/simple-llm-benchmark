from llm_benchmark import inspect_provider


def test_provider_passes_verify_false_to_httpx(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_client(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(inspect_provider, "OpenAIAsyncHttpxClient", fake_client)
    provider = object.__new__(inspect_provider.TLSOpenAICompatibleAPI)
    provider.tls_verify = False
    provider.client_timeout = None
    assert provider._create_http_client() is sentinel
    assert captured["verify"] is False
