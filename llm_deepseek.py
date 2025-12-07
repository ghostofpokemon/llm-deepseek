import llm
from llm.default_plugins.openai_models import Chat, Completion
from llm.utils import remove_dict_none_values
from pathlib import Path
import json
import time
import httpx
import os
from typing import Optional
from pydantic import Field

try:
    from rich.console import Console
    from rich.style import Style
except Exception:  # rich is optional; fall back to plain output
    Console = None
    Style = None

# Constants for cache timeout and API base URL
CACHE_TIMEOUT = 3600
DEFAULT_API_BASE = "https://api.deepseek.com"
DEEPSEEK_API_BASE = os.environ.get("LLM_DEEPSEEK_BASE_URL", DEFAULT_API_BASE)
DEEPSEEK_BETA_API_BASE = os.environ.get(
    "LLM_DEEPSEEK_BETA_BASE_URL", f"{DEFAULT_API_BASE}/beta"
)
# Speciale endpoint is available out of the box; no env config required.
DEEPSEEK_SPECIALE_BASE = "https://api.deepseek.com/v3.2_speciale_expires_on_20251215"
DEEPSEEK_MODELS_URL = os.environ.get(
    "LLM_DEEPSEEK_MODELS_URL", "https://api.deepseek.com/models"
)

def get_deepseek_models():
    """Fetch and cache DeepSeek models."""
    key = llm.get_key("", "deepseek", "LLM_DEEPSEEK_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else None
    return fetch_cached_json(
        url=DEEPSEEK_MODELS_URL,
        path=llm.user_dir() / "deepseek_models.json",
        cache_timeout=CACHE_TIMEOUT,
        headers=headers
    )["data"]

def get_model_ids_with_aliases(models):
    """Extract model IDs and create empty aliases list."""
    return [(model['id'], []) for model in models]

class DeepSeekChat(Chat):
    needs_key = "deepseek"
    key_env_var = "LLM_DEEPSEEK_KEY"

    def __init__(self, model_id, api_base=None, **kwargs):
        super().__init__(model_id, **kwargs)
        self.api_base = api_base or DEEPSEEK_API_BASE
        self.console = Console() if Console else None
        self.reasoning_style = Style(color="cyan", dim=True, italic=True) if Style else None

    def __str__(self):
        return f"DeepSeek Chat: {self.model_id}"

    class Options(Chat.Options):
        prefill: Optional[str] = Field(
            description="Initial text for the model's response (beta feature). Uses DeepSeek's Chat Prefix Completion.",
            default=None
        )
        response_format: Optional[str] = Field(
            description="Format of the response (e.g., 'json_object').",
            default=None
        )

    def execute(self, prompt, stream, response, conversation, key=None):
        messages = self._build_messages(conversation, prompt)
        response._prompt_json = {"messages": messages}
        kwargs = remove_dict_none_values(self.build_kwargs(prompt, stream))

        max_tokens = kwargs.pop('max_tokens', 8192)
        if prompt.options.response_format:
            kwargs["response_format"] = {"type": prompt.options.response_format}

        # Remove options that aren't supported by the OpenAI client
        kwargs.pop('prefill', None)
        kwargs.pop('show_reasoning', None)

        client = self.get_client(key)

        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                **kwargs,
            )

            if stream:
                yield from self._stream_completion(completion)
            else:
                yield from self._non_stream_completion(completion)

            response.response_json = {"content": "".join(response._chunks)}

            # Store reasoning_content in response if available
            if not stream and hasattr(completion.choices[0].message, "reasoning_content"):
                response.response_json["reasoning_content"] = completion.choices[0].message.reasoning_content
                
        except httpx.HTTPError as e:
            raise llm.ModelError(f"DeepSeek API error: {str(e)}")

    def _stream_completion(self, completion):
        reasoning_started = False

        for chunk in completion:
            delta = chunk.choices[0].delta

            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                if self.console and not reasoning_started:
                    self.console.print("\n[Reasoning]\n\n", style=self.reasoning_style, end="")
                    reasoning_started = True
                if self.console:
                    self.console.print(reasoning_content, style=self.reasoning_style, end="")
                else:
                    yield reasoning_content

            content = delta.content
            if content:
                if self.console and reasoning_started:
                    self.console.print("\n[Response]\n\n", style="bold green", end="")
                    reasoning_started = False
                yield content

    def _non_stream_completion(self, completion):
        message = completion.choices[0].message

        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning = message.reasoning_content
            if self.console:
                self.console.print("\n[Reasoning]\n\n", style=self.reasoning_style)
                self.console.print(reasoning, style=self.reasoning_style)
                self.console.print("\n[Response]\n\n", style="bold green", end="")
            else:
                yield reasoning
                yield "\n\n"

        content = message.content
        if content:
            yield content

    def _build_messages(self, conversation, prompt):
        """Build the messages list for the API call."""
        messages = []
        if conversation:
            for prev_response in conversation.responses:
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})

        # Add system message if provided
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        messages.append({"role": "user", "content": prompt.prompt})

        if prompt.options.prefill:
            prefill_content = prompt.options.prefill
            # Check if prefill value is a file path
            if os.path.exists(prefill_content) and os.path.isfile(prefill_content):
                try:
                    with open(prefill_content, 'r') as file:
                        prefill_content = file.read()
                except Exception as e:
                    print(f"Warning: Could not read prefill file '{prompt.options.prefill}': {e}")
            
            messages.append({
                "role": "assistant",
                "content": prefill_content,
                "prefix": True
            })

        return messages

class DeepSeekCompletion(Completion):
    needs_key = "deepseek"
    key_env_var = "LLM_DEEPSEEK_KEY"

    def __init__(self, model_id, api_base=None, **kwargs):
        super().__init__(model_id, **kwargs)
        # Text completions (including FIM beta) still run on the beta endpoint
        # so we expose a separate override to keep compatibility.
        self.api_base = api_base or DEEPSEEK_BETA_API_BASE

    def __str__(self):
        return f"DeepSeek Completion: {self.model_id}"

    class Options(Completion.Options):
        prefill: Optional[str] = Field(
            description="Initial text for the model's response (beta feature). Uses DeepSeek's Completion Prefix.",
            default=None
        )
        echo: Optional[bool] = Field(
            description="Echo back the prompt in addition to the completion.",
            default=None
        )

    def execute(self, prompt, stream, response, conversation, key=None):
        full_prompt = self._build_full_prompt(conversation, prompt)
        response._prompt_json = {"prompt": full_prompt}
        kwargs = remove_dict_none_values(self.build_kwargs(prompt, stream))

        max_tokens = kwargs.pop('max_tokens', 4096)
        if prompt.options.echo:
            kwargs["echo"] = prompt.options.echo

        # Remove custom options from kwargs
        kwargs.pop('prefill', None)
        kwargs.pop('show_reasoning', None)  # Remove if it exists
        kwargs.pop('thinking', None)

        client = self.get_client(key)

        try:
            completion = client.completions.create(
                model=self.model_name,
                prompt=full_prompt,
                stream=stream,
                max_tokens=max_tokens,
                **kwargs,
            )

            for chunk in completion:
                text = chunk.choices[0].text
                if text:
                    yield text

            response.response_json = {"content": "".join(response._chunks)}
        except httpx.HTTPError as e:
            raise llm.ModelError(f"DeepSeek API error: {str(e)}")

    def _build_full_prompt(self, conversation, prompt):
        """Build the full prompt for the API call."""
        messages = []
        if conversation:
            for prev_response in conversation.responses:
                messages.append(prev_response.prompt.prompt)
                messages.append(prev_response.text())
        messages.append(prompt.prompt)

        # Include system message if provided
        if prompt.system:
            messages.insert(0, prompt.system)

        full_prompt = "\n".join(messages)
        if prompt.options.prefill:
            prefill_content = prompt.options.prefill
            # Check if prefill value is a file path
            if os.path.exists(prefill_content) and os.path.isfile(prefill_content):
                try:
                    with open(prefill_content, 'r') as file:
                        prefill_content = file.read()
                except Exception as e:
                    print(f"Warning: Could not read prefill file '{prompt.options.prefill}': {e}")
            
            full_prompt += f"\n{prefill_content}"

        return full_prompt

class DownloadError(Exception):
    pass

def fetch_cached_json(url, path, cache_timeout, headers=None):
    """Fetch JSON data from a URL and cache it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and time.time() - path.stat().st_mtime < cache_timeout:
        with open(path, "r") as file:
            return json.load(file)

    try:
        response = httpx.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx.HTTPError:
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            raise DownloadError(f"Failed to download data and no cache is available at {path}")

@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "deepseek", "LLM_DEEPSEEK_KEY")
    if not key:
        return
    try:
        models = get_deepseek_models()
        models_with_aliases = get_model_ids_with_aliases(models)
        
        # First register all Chat models
        for model_id, aliases in models_with_aliases:
            register(
                DeepSeekChat(
                    model_id=f"deepseekchat/{model_id}",
                    model_name=model_id,
                ),
                aliases=[model_id]
            )

        # Register the Speciale thinking-only variant explicitly with a clear name
        register(
            DeepSeekChat(
                model_id="deepseekchat/deepseek-reasoner-speciale",
                model_name="deepseek-reasoner",
                api_base=DEEPSEEK_SPECIALE_BASE,
            ),
            aliases=["deepseek-reasoner-speciale"]
        )
        
        # Then register Completion models (excluding reasoner)
        for model_id, aliases in models_with_aliases:
            if "reasoner" not in model_id.lower():
                register(
                    DeepSeekCompletion(
                        model_id=f"deepseekcompletion/{model_id}",
                        model_name=model_id,
                    ),
                    aliases=[f"{model_id}-completion"]
                )
    except DownloadError as e:
        print(f"Error fetching DeepSeek models: {e}")

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    def deepseek_models():
        """List available DeepSeek models."""
        key = llm.get_key("", "deepseek", "LLM_DEEPSEEK_KEY")
        if not key:
            print("DeepSeek API key not set. Use 'llm keys set deepseek' to set it.")
            return
        try:
            models = get_deepseek_models()
            models_with_aliases = get_model_ids_with_aliases(models)
            
            # First display all Chat models
            for model_id, aliases in models_with_aliases:
                print(f"DeepSeek Chat: deepseekchat/{model_id}")
                print(f"  Aliases: {model_id}")
                print()

            # Speciale thinking-only model
            print("DeepSeek Chat: deepseekchat/deepseek-reasoner-speciale")
            print(f"  Aliases: deepseek-reasoner-speciale")
            print(f"  Base URL: {DEEPSEEK_SPECIALE_BASE}")
            print()
            
            # Then display all Completion models (excluding reasoner)
            for model_id, aliases in models_with_aliases:
                if "reasoner" not in model_id.lower():
                    print(f"DeepSeek Completion: deepseekcompletion/{model_id}")
                    print(f"  Aliases: {model_id}-completion")
                    print()
        except DownloadError as e:
            print(f"Error fetching DeepSeek models: {e}")
