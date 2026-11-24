"""Test the LLM judge that the judged metrics ask."""

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kgeval.errors import JudgeError, VerdictError
from kgeval.judge import Judge
from kgeval.metrics.judged.facts import SupportVerdict

VERDICT = SupportVerdict(reason="Stated.", supported=True)


class ChatServer(ThreadingHTTPServer):
    """Answer chat completions with the scripted error statuses, then with ``content``."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ChatHandler)
        self.requests: list[dict[str, Any]] = []
        self.statuses: list[int] = []
        self.content = VERDICT.model_dump_json()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class ChatHandler(BaseHTTPRequestHandler):
    server: ChatServer

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        status = self.server.statuses.pop(0) if self.server.statuses else 200
        reply: dict[str, Any] = {"error": {"message": f"status {status}", "type": "error"}}
        if status == 200:
            message = {"role": "assistant", "content": self.server.content}
            reply = {
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            }
        data = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return None


@pytest.fixture
def server() -> Iterator[ChatServer]:
    chat_server = ChatServer()
    thread = threading.Thread(target=chat_server.serve_forever, daemon=True)
    thread.start()
    yield chat_server
    chat_server.shutdown()
    chat_server.server_close()


def judge_for(server: ChatServer, **settings: Any) -> Judge:
    return Judge("openai/test-model", base_url=server.url, api_key="unused", **settings)


def test_judge_asks_for_the_answer_schema_and_validates_the_answer(server: ChatServer) -> None:
    judge = judge_for(server, temperature=0)

    assert judge.ask("Is it so?", SupportVerdict) == VERDICT
    [request] = server.requests
    assert request["model"] == "test-model"
    assert request["messages"] == [{"role": "user", "content": "Is it so?"}]
    assert request["response_format"]["json_schema"]["schema"]["required"] == [
        "reason",
        "supported",
    ]
    assert request["temperature"] == 0
    # Name the settings, so the cache and signatures tell judges with other settings apart.
    assert judge.name == "openai/test-model|temperature=0"


@pytest.mark.parametrize(("status", "content"), [(400, ""), (200, '{"verdict": "yes"}')])
def test_judge_reports_rejected_prompts_and_invalid_answers_as_unusable_verdicts(
    server: ChatServer, status: int, content: str
) -> None:
    server.statuses = [status] if status != 200 else []
    server.content = content

    with pytest.raises(VerdictError, match="could not judge a prompt"):
        judge_for(server).ask("Is it so?", SupportVerdict)
    assert len(server.requests) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_judge_retries_rate_limits_and_server_errors(
    server: ChatServer, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    server.statuses = [status] * 5

    assert judge_for(server, max_retries=0).ask("Is it so?", SupportVerdict) == VERDICT
    assert len(server.requests) == 6


def test_judge_wraps_other_api_errors_as_failures_of_the_judge(server: ChatServer) -> None:
    server.statuses = [401]

    with pytest.raises(JudgeError, match="request failed") as raised:
        judge_for(server).ask("Is it so?", SupportVerdict)
    assert not isinstance(raised.value, VerdictError)


def test_judge_rejects_a_model_without_a_provider() -> None:
    with pytest.raises(JudgeError, match="unknown judge model gpt5; name it as provider/model"):
        Judge("gpt5")


def test_cached_verdicts_are_reused_across_instances(server: ChatServer, tmp_path: Path) -> None:
    cache = tmp_path / "verdicts.sqlite"

    for prompt in ("Is it so?", "Is it so?", "Is it not?"):
        assert judge_for(server, cache=cache).ask(prompt, SupportVerdict) == VERDICT

    assert [request["messages"][0]["content"] for request in server.requests] == [
        "Is it so?",
        "Is it not?",
    ]


def test_invalid_cached_verdicts_are_judged_again_and_replaced(
    server: ChatServer, tmp_path: Path
) -> None:
    judge = judge_for(server, cache=tmp_path / "verdicts.sqlite")
    judge.ask("Is it so?", SupportVerdict)
    with closing(sqlite3.connect(tmp_path / "verdicts.sqlite")) as connection, connection:
        connection.execute("UPDATE verdicts SET answer = '{}'")

    judge.ask("Is it so?", SupportVerdict)
    judge.ask("Is it so?", SupportVerdict)

    assert len(server.requests) == 2


def test_an_unusable_cache_file_raises_a_judge_error(server: ChatServer, tmp_path: Path) -> None:
    with pytest.raises(JudgeError, match="cannot use the judge cache"):
        judge_for(server, cache=tmp_path)
