# Test2

## Prerequisites

1. Configure `.env`:

```bash
copy .env.sample .env
```

And fill all required values

2. Add data.csv to the data directory

## Installation

Make sure that ollama is installed on your machine, or run it with docker compose:

```bash
docker volume create ollama_data  # We use external volumes in docker for persistency (aka one of my best practices)
docker compose up -d
```

## Running

Run app with uv `the best python package and project manager`:

```bash
uv sync --frozen --no-dev --compile-bytecode  # Create .venv and compile bytecode for faster inference
uv run --frozen --no-dev app/main.py
```

## Why Ollama and func calls

First of all - CSV is a structured data, so no embeddings is necessary (as its typically used for unstructured texts)
Secondly, tool call is used for calculations.
Lastly, I used ollama, because I prefer to run my computations on my hardware, but if needed larger LLMs, I stick with openrouter, as it gives me flexible choice of proprietary models and providers as well.
In terms of models I used qwen3, as its thinking model, which means better results in general, and it supports tool calls natively as well.
The only downside is that, thinking models can be slow, but it leveraged by its higher accuracy.

## Considerations

Main function is used for calculating all values, so the model needs only to analyze them and answer user's question based on that.
Pandas is used for CSV data load. 
Pydantic settings is used for .env handling.

That is the most efficient solution that I found so far, and as I see even small open source models (8 billion parameters) are capable of solving that task.
The reason for that is because we abstract away all LLM's magic, and delegate all calculations to the tools, avoiding ambiguity for letting LLM calculate everything by itself.