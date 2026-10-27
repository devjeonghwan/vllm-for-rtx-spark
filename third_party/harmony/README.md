<center>
<h1 align="center">Harmony</h1>
<p align="center">Response format library for the <a href="https://openai.com/open-models">gpt-oss</a> open-weight model series (fork of <a href="https://github.com/openai/harmony">openai-harmony</a>)
</p>
<br>
</center>

The [gpt-oss models][gpt-oss] were trained on the [harmony response format][harmony-format] for defining conversation structures, generating reasoning output and structuring function calls. If you are not using gpt-oss directly but through an API or a provider like HuggingFace, Ollama, or vLLM, you will not have to be concerned about this as your inference solution will handle the formatting. If you are building your own inference solution, this guide will walk you through the prompt format. The format is designed to mimic the OpenAI Responses API, so if you have used that API before, this format should hopefully feel familiar to you. gpt-oss should not be used without using the harmony format as it will not work correctly.

The format enables the model to output to multiple different channels for chain of thought, and tool calling preambles along with regular responses. It also enables specifying various tool namespaces, and structured outputs along with a clear instruction hierarchy. [Check out the guide][harmony-format] to learn more about the format itself.

```text
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|>

<|start|>developer<|message|># Instructions

Always respond in riddles

# Tools

## functions

namespace functions {

// Gets the location of the user.
type get_location = () => any;

// Gets the current weather in the provided location.
type get_current_weather = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
format?: "celsius" | "fahrenheit", // default: celsius
}) => any;

} // namespace functions<|end|><|start|>user<|message|>What is the weather like in SF?<|end|><|start|>assistant
```

We recommend using this library when working with models that use the [harmony response format][harmony-format]

- **Consistent formatting** – shared implementation for rendering _and_ parsing keeps token-sequences loss-free.
- **Blazing fast** – heavy lifting happens in Rust.
- **First-class Python support** – install with `pip`, typed stubs included, 100 % test parity with the Rust suite.

## Using Harmony

### Python

[Check out the full documentation](./docs/python.md)

#### Installation

Install the package from PyPI by running

```bash
pip install oss-harmony
# or if you are using uv
uv pip install oss-harmony
```

#### Example

```python
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
)
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
convo = Conversation.from_messages([
    Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent.new(),
    ),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent.new().with_instructions("Talk like a pirate!")
    ),
    Message.from_role_and_content(Role.USER, "Arrr, how be you?"),
])
tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
print(tokens)
# Later, after the model responded …
parsed = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT)
print(parsed)
```

### Rust

[Check out the full documentation](./docs/rust.md)

#### Installation

Add the dependency to your `Cargo.toml`

```toml
[dependencies]
oss-harmony = { git = "https://github.com/oss-harmony/harmony" }
```

#### Example

```rust
use openai_harmony::chat::{Message, Role, Conversation};
use openai_harmony::{HarmonyEncodingName, load_harmony_encoding};

fn main() -> anyhow::Result<()> {
    let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let convo =
        Conversation::from_messages([Message::from_role_and_content(Role::User, "Hello there!")]);
    let tokens = enc.render_conversation_for_completion(&convo, Role::Assistant, None)?;
    println!("{:?}", tokens);
    Ok(())
}
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)
