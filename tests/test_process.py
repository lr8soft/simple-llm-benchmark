from llm_benchmark.process import utf8_subprocess_env


def test_inspect_children_are_forced_to_utf8() -> None:
    env = utf8_subprocess_env({"EXISTING": "kept", "PYTHONUTF8": "0"})
    assert env["EXISTING"] == "kept"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
