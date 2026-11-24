# LLM judges

Some qualities cannot be checked by rules, such as whether two names refer to the same
thing or whether a chunk actually states a fact. The six
[judged metrics](metrics.md#judged-metrics) ask an LLM instead. Any model that
[LiteLLM](https://docs.litellm.ai/) supports can judge, from a hosted provider or a
local server.

## Choose a model

Name the model as LiteLLM does, `provider/model`. The API key comes from the provider's
environment variable, such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. The
[LiteLLM provider list](https://docs.litellm.ai/docs/providers) gives each provider's
model names and environment variables.

=== "CLI"

    ```console
    export OPENAI_API_KEY=<api-key>
    kgeval run --judge openai/gpt-5-mini
    ```

=== "Python"

    ```python
    judge = kgeval.Judge("openai/gpt-5-mini")
    metrics = kgeval.default_metrics(judge=judge)
    report = kgeval.evaluate(graph, metrics)
    ```

When the judge gives no usable verdict for one prompt, because the model declines, its
answer does not parse, or the server rejects the prompt, kgeval asks once more and
otherwise leaves that unit out. The report counts such units as unjudged, and the score
then carries a confidence interval, since the judged units no longer cover the
population. Requests that hit a rate limit or a server error are retried for up to
about a minute. If the judge fails otherwise, for example because the API rejects the
key, or gives no usable verdict at all, the metric is skipped with the error as its
reason, and the other metrics still run. Each finding includes the judge's reason for
its verdict.

### Request settings

In Python, other keyword arguments of `Judge` go into every request, such as a
temperature or a seed for more repeatable verdicts, or a reasoning effort:

```python
judge = kgeval.Judge("openai/gpt-5-mini", reasoning_effort="low")
```

The settings become part of the judge's name, `openai/gpt-5-mini|reasoning_effort="low"`,
so the cache and metric signatures tell judges with different settings apart.

## Local models

Local servers need no API key. Give their base URL with `--judge-base-url`, or with the
provider's base URL variable, such as `HOSTED_VLLM_API_BASE`. Any other
OpenAI-compatible server works as the `hosted_vllm` provider when it needs no key, or
as the `openai` provider with its key in `OPENAI_API_KEY`:

```console
kgeval run --judge hosted_vllm/<model> --judge-base-url <base-url>
```

Small models judge most reliably when their JSON verdicts contain no extra whitespace
and no thinking. With vLLM:

```console
vllm serve Qwen/Qwen3-8B \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'
kgeval run --judge hosted_vllm/Qwen/Qwen3-8B --judge-base-url <base-url>
```

## Sampling

All judged metrics except `vocabulary_resolution` judge a sample of at most 100 units,
so the cost stays bounded however large the graph is. `vocabulary_resolution` judges the whole
vocabulary, showing each term with how the graph uses it and a few grounded examples,
at most 200 terms per call. A larger vocabulary is split into calls that each list the
100 most used terms, so a rare synonym of a common term is still found, and a window of
the other terms sorted by usage, so that rare terms used alike share a call.

The sample is reproducible. It is chosen by hashing the content of each unit with a
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
    kgeval run --judge openai/gpt-5-mini --judge-cache verdicts.sqlite
    ```

=== "Python"

    ```python
    judge = kgeval.Judge("openai/gpt-5-mini", cache="verdicts.sqlite")
    ```

Verdicts are keyed by the judge's name, the prompt, and the expected answer format, so a
different model or prompt never reuses them. Delete the file to judge again.
