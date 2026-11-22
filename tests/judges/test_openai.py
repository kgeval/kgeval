"""Test judging with a model served through the OpenAI API."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, cast

import openai
import pytest

from kgeval.errors import JudgeError, VerdictError
from kgeval.judges.base import Judge
from kgeval.judges.openai import OpenAIJudge
from kgeval.metrics.judged.facts import SupportVerdict


class FakeCompletions:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> object:
        return self._answer(kwargs)

    def create(self, **kwargs: Any) -> object:
        return self._answer(kwargs)

    def _answer(self, kwargs: dict[str, Any]) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def judge_with(
    completions: FakeCompletions,
    *,
    json_schema: bool = True,
    request_body: dict[str, object] | None = None,
) -> OpenAIJudge:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIJudge(
        "test-model",
        json_schema=json_schema,
        client=cast(openai.OpenAI, client),
        request_body=request_body,
    )


def completion(**message: object) -> SimpleNamespace:
    fields: dict[str, object] = {"refusal": None, "parsed": None, "content": None}
    choice = SimpleNamespace(message=SimpleNamespace(**(fields | message)), finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def test_judge_returns_the_parsed_structured_answer() -> None:
    verdict = SupportVerdict(reason="Stated.", supported=True)
    completions = FakeCompletions(completion(parsed=verdict))

    assert judge_with(completions).ask("Is it so?", SupportVerdict) is verdict
    [call] = completions.calls
    assert call["model"] == "test-model"
    assert call["response_format"] is SupportVerdict
    assert call["messages"] == [{"role": "user", "content": "Is it so?"}]


def test_judge_reports_refusals_as_unusable_verdicts() -> None:
    completions = FakeCompletions(completion(refusal="I cannot help with that."))

    with pytest.raises(VerdictError, match="declined to judge: I cannot help with that"):
        judge_with(completions).ask("Is it so?", SupportVerdict)


def test_judge_reports_missing_answers_as_unusable_verdicts() -> None:
    with pytest.raises(VerdictError, match="gave no SupportVerdict"):
        judge_with(FakeCompletions(completion())).ask("Is it so?", SupportVerdict)


def test_judge_reports_prompts_the_server_rejects_as_unusable_verdicts() -> None:
    # Fake the response, so the test does not depend on the SDK's HTTP library.
    response = SimpleNamespace(request=None, status_code=400, headers={})
    error = openai.BadRequestError(
        "context length exceeded", response=cast(Any, response), body=None
    )

    with pytest.raises(VerdictError, match="could not judge a prompt: context length exceeded"):
        judge_with(FakeCompletions(error=error)).ask("Is it so?", SupportVerdict)


def test_judge_wraps_other_api_errors_as_failures_of_the_judge() -> None:
    error = openai.OpenAIError("connection refused")

    with pytest.raises(JudgeError, match="request failed") as raised:
        judge_with(FakeCompletions(error=error)).ask("Is it so?", SupportVerdict)
    assert not isinstance(raised.value, VerdictError)


def test_request_body_fields_go_into_every_request_and_the_judges_name() -> None:
    verdict = SupportVerdict(reason="Stated.", supported=True)
    completions = FakeCompletions(completion(parsed=verdict))
    judge = judge_with(completions, request_body={"temperature": 0, "seed": 7})

    judge.ask("Is it so?", SupportVerdict)

    assert completions.calls[0]["extra_body"] == {"temperature": 0, "seed": 7}
    # Name the settings, so the cache and signatures tell judges with other settings apart.
    assert judge.name == "test-model|seed=7|temperature=0"


def test_json_object_mode_puts_the_schema_in_the_prompt_and_validates_the_answer() -> None:
    answer = json.dumps({"reason": "Stated.", "supported": True})
    completions = FakeCompletions(completion(content=answer))

    verdict = judge_with(completions, json_schema=False).ask("Is it so?", SupportVerdict)

    assert verdict == SupportVerdict(reason="Stated.", supported=True)
    [call] = completions.calls
    assert call["response_format"] == {"type": "json_object"}
    assert '"supported"' in call["messages"][0]["content"]
    assert call["messages"][1] == {"role": "user", "content": "Is it so?"}


def test_json_object_mode_rejects_answers_that_break_the_schema() -> None:
    completions = FakeCompletions(completion(content='{"verdict": "yes"}'))

    with pytest.raises(VerdictError, match="gave an invalid SupportVerdict"):
        judge_with(completions, json_schema=False).ask("Is it so?", SupportVerdict)


def test_judge_without_an_api_key_fails_on_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(JudgeError, match="cannot create a client for test-model"):
        OpenAIJudge("test-model")


def test_judge_satisfies_the_judge_protocol() -> None:
    judge = judge_with(FakeCompletions())

    assert isinstance(judge, Judge)
    assert judge.name == "test-model"


class ChatServer(ThreadingHTTPServer):
    """Answer every chat completion with one fixed verdict, recording each request."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ChatHandler)
        self.requests: list[dict[str, Any]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class ChatHandler(BaseHTTPRequestHandler):
    server: ChatServer

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        content = json.dumps({"reason": "Stated.", "supported": True})
        reply = json.dumps(
            {
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

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


@pytest.mark.parametrize(
    ("json_schema", "format_type"), [(True, "json_schema"), (False, "json_object")]
)
def test_judge_round_trips_through_the_openai_sdk(
    server: ChatServer, json_schema: bool, format_type: str
) -> None:
    judge = OpenAIJudge(
        "local-model", base_url=server.url, api_key="unused", json_schema=json_schema
    )

    verdict = judge.ask("Is it so?", SupportVerdict)

    assert verdict == SupportVerdict(reason="Stated.", supported=True)
    [request] = server.requests
    assert request["model"] == "local-model"
    assert request["response_format"]["type"] == format_type
