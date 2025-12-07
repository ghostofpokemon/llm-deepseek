# llm-deepseek

[![PyPI](https://img.shields.io/pypi/v/llm-deepseek-xtreme.svg)](https://pypi.org/project/llm-deepseek-xtreme/)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ghostofpokemon/llm-deepseek?include_prereleases)](https://github.com/ghostofpokemon/llm-deepseek/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ghostofpokemon/llm-deepseek/blob/main/LICENSE)

LLM access to DeepSeek's API

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-deepseek-xtreme
```

## Usage

First, set an [API key](https://platform.deepseek.com/api_keys) for DeepSeek:

```bash
llm keys set deepseek
# Paste key here
```

### Models (DeepSeek-V3.2)

- `deepseek-chat`: V3.2 non-thinking mode, 128K context, json output + tool calls + chat prefix completion
- `deepseek-reasoner`: V3.2 thinking mode, 128K context, json output + tool calls + chat prefix completion; reasoning always shown
- `deepseek-reasoner-speciale`: V3.2-Speciale thinking-only endpoint at `https://api.deepseek.com/v3.2_speciale_expires_on_20251215`, up to 128K output, no tools/json/prefix completion (auto-registered)

Chat/reasoner calls use `https://api.deepseek.com` (Speciale uses its own base automatically). Text completions/FIM automatically use the beta endpoint `https://api.deepseek.com/beta`.

Pricing (from https://api-docs.deepseek.com/quick_start/pricing): $0.028/M input tokens (cache hit), $0.28/M input tokens (cache miss), $0.42/M output tokens.

Run `llm models` to list the models, and `llm models --options` to include a list of their options.

### Running Prompts

Run prompts like this:

```bash
llm -m deepseek-chat "Describe a futuristic city on Mars"
llm -m deepseek-chat-completion "The AI began to dream, and in its dreams," -o echo true
llm -m deepseek-reasoner "Write a Python function to sort a list of numbers"
```

Note: The DeepSeek Reasoner model only supports the chat endpoint, not the completion endpoint.

### DeepSeek Reasoner Model

The DeepSeek Reasoner model uses a Chain of Thought (CoT) approach to solve complex problems, showing its reasoning process before providing the final answer.

The plugin always shows the model's chain of thought reasoning in non-streaming mode. The reasoning feature is currently only supported in non-streaming mode.

```bash
# Normal usage - will show reasoning by default
llm -m deepseek-reasoner "What is 537 * 943?"
```

### New Features

#### Prefill

The `prefill` option allows you to provide initial text for the model's response. This is useful for guiding the model's output.

Example:

```bash
llm -m deepseek-chat "What are some wild and crazy activities for a holiday party?" -o prefill "Here are some off-the-wall ideas to make your holiday party unforgettable [warning: these may not be suitable for work holiday parties]:"
```

You can also load prefill text from a file:

```bash
# Create a file with your prefill text
echo "Here are some unique holiday party ideas:" > prefill.txt

# Use the file path as the prefill value
llm -m deepseek-chat "What are some fun activities for a holiday party?" -o prefill prefill.txt
```

This is especially useful for longer prefill text that would be unwieldy on the command line.

#### JSON Response Format

The `response_format` option allows you to specify that the model should output its response in JSON format. To ensure the model outputs valid JSON, include the word "json" in the system or user prompt. Optionally, you can provide an example of the desired JSON format to guide the model.

Example:

```bash
llm -m deepseek-chat "What are some fun activities for a holiday party?" -o response_format json_object --system "json"
```

To guide the model further, you can provide an example JSON structure:

```bash
llm -m deepseek-chat "What are some way to tell if a holiday party is fun?" -o response_format json_object --system 'EXAMPLE JSON OUTPUT: {"event": "holiday_party_fun", "success_metric": ["..."]}'
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-deepseek
python3 -m venv venv
source venv/bin/activate
```

Publish to PyPI after bumping the version:

```bash
pip install -e '.[test]' build twine
python -m build
twine upload dist/*
```
