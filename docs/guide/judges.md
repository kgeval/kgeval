# LLM judges

Some qualities can't be checked by rules, such as whether two names refer to the same
thing or whether a chunk actually states a fact. The six
[judged metrics](metrics.md#judged-metrics) ask an LLM instead. Any model served
through the OpenAI API can judge, from a hosted provider or a local server.

## Choose a model

Name the model as `provider:model`. A model ID without a provider prefix is an OpenAI
model.

=== "CLI"

    ```console
    export OPENAI_API_KEY=sk-...
    kgeval run --judge openai:gpt-5-mini
    ```

=== "Python"

    ```python
    judge = kgeval.create_judge("openai:gpt-5-mini")
    metrics = kgeval.default_metrics(judge=judge)
    report = kgeval.evaluate(graph, metrics)
    ```

When the judge gives no usable verdict for one prompt, because the model declines, its
answer doesn't parse or the server rejects the prompt, kgeval asks once more and
otherwise leaves that unit out. The report counts such units as unjudged, and the score
then carries a confidence interval, since the judged units no longer cover the
population. If the judge fails otherwise, for example because the API rejects the key,
or gives no usable verdict at all, the metric is skipped with the error as its reason,
and the other metrics still run. Each finding includes the judge's reason for its
verdict.

### Request settings

In Python, `request_body` adds fields to every request, such as a temperature or a
seed for more repeatable verdicts, or a reasoning effort:

```python
judge = kgeval.create_judge("openai:gpt-5-mini", request_body={"reasoning_effort": "low"})
```

The fields become part of the judge's name, `openai:gpt-5-mini|reasoning_effort="low"`,
so the cache and metric signatures tell judges with different settings apart.

## Providers

Each provider reads its base URL and API key from `<PREFIX>_BASE_URL` and
`<PREFIX>_API_KEY`. The base URL variable is optional when the provider has a default.

| Provider      | Model              | Prefix         | Default base URL                                           |
| ------------- | ------------------ | -------------- | ---------------------------------------------------------- |
| OpenAI        | `openai:MODEL`     | `OPENAI`       | `https://api.openai.com/v1`                                |
| Azure OpenAI  | `azure:DEPLOYMENT` | `AZURE_OPENAI` | None                                                       |
| Google Gemini | `gemini:MODEL`     | `GEMINI`       | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Mistral       | `mistral:MODEL`    | `MISTRAL`      | `https://api.mistral.ai/v1`                                |
| xAI           | `xai:MODEL`        | `XAI`          | `https://api.x.ai/v1`                                      |
| DeepSeek      | `deepseek:MODEL`   | `DEEPSEEK`     | `https://api.deepseek.com`                                 |
| Groq          | `groq:MODEL`       | `GROQ`         | `https://api.groq.com/openai/v1`                           |
| Together AI   | `together:MODEL`   | `TOGETHER`     | `https://api.together.xyz/v1`                              |
| Fireworks AI  | `fireworks:MODEL`  | `FIREWORKS`    | `https://api.fireworks.ai/inference/v1`                    |
| OpenRouter    | `openrouter:MODEL` | `OPENROUTER`   | `https://openrouter.ai/api/v1`                             |
| Ollama        | `ollama:MODEL`     | `OLLAMA`       | `http://localhost:11434/v1`                                |
| LM Studio     | `lmstudio:MODEL`   | `LMSTUDIO`     | `http://localhost:1234/v1`                                 |
| vLLM          | `vllm:MODEL`       | `VLLM`         | `http://localhost:8000/v1`                                 |

Azure OpenAI has no default base URL. Set it to your resource's endpoint:

```console
export AZURE_OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/v1/
export AZURE_OPENAI_API_KEY=...
kgeval run --judge azure:<deployment>
```

## Local models

Ollama, LM Studio, and vLLM need no API key. To reach a server on another host or port,
set its base URL variable, such as `VLLM_BASE_URL`, or pass `--judge-base-url`. For any
other OpenAI-compatible server that needs no key, use one of these prefixes with
`--judge-base-url`:

```console
kgeval run --judge vllm:my-model --judge-base-url http://gpu-box:9000/v1
```

Small models judge most reliably when their JSON verdicts contain no extra whitespace
and no thinking. With vLLM:

```console
vllm serve Qwen/Qwen3-8B \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'
kgeval run --judge vllm:Qwen/Qwen3-8B
```

## Sampling

All judged metrics except `vocabulary_resolution` judge a sample of at most 100 units,
so the cost stays bounded however large the graph is. `vocabulary_resolution` judges the whole
vocabulary, showing each term with how the graph uses it and a few grounded examples,
at most 200 terms per call. A larger vocabulary is split into calls that each list the
100 most used terms, so a rare synonym of a common term is still found, and a window of
the other terms sorted by usage, so that rare terms used alike share a call.

The sample is reproducible. It's chosen by hashing the content of each unit with a
seed, so the same graph gets the same sample on every run and platform, and so does a
rebuilt graph with the same content. When the sample is smaller than the population,
the score comes with a 95% Wilson score interval.

| Option          | Default | Description                                    |
| --------------- | ------- | ---------------------------------------------- |
| `--sample-size` | 100     | The most units each sampled metric judges.     |
| `--seed`        | 0       | The seed that selects which units are sampled. |

In Python, pass `sample_size` and `seed` to
[`default_metrics`][kgeval.default_metrics].

## Caching

A cache keeps every verdict in a SQLite file, so a rerun gives the same scores without
calling the model again:

=== "CLI"

    ```console
    kgeval run --judge openai:gpt-5-mini --judge-cache verdicts.sqlite
    ```

=== "Python"

    ```python
    judge = kgeval.CachedJudge(kgeval.create_judge("openai:gpt-5-mini"), "verdicts.sqlite")
    ```

Verdicts are keyed by the judge's name, the prompt, and the expected answer format, so a
different model or prompt never reuses them. Delete the file to judge again.

## Custom judges

Any object with a `name` and an `ask(prompt, output_type)` method can judge. `ask`
must return an instance of `output_type`, a Pydantic model, and be safe to call from
several threads at once:

```python
class MyJudge:
    name = "my-model"

    def ask(self, prompt, output_type):
        reply = my_client.generate(prompt, schema=output_type.model_json_schema())
        return output_type.model_validate_json(reply)


metrics = kgeval.default_metrics(judge=MyJudge())
```

The `name` appears in every judged metric's signature, so reports from different judges
are never mistaken for each other. When one prompt gets no usable answer, raise
[`VerdictError`][kgeval.VerdictError], so that the metric leaves that unit out and
judges the rest; to signal any other failure, raise [`JudgeError`][kgeval.JudgeError].
