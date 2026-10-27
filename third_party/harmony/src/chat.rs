use core::fmt;
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize,
};
use std::collections::BTreeMap;
use std::{fmt::Display, marker::PhantomData};

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Author {
    pub role: Role,
    pub name: Option<String>,
}

impl Author {
    pub fn new(role: Role, name: impl Into<String>) -> Self {
        Self {
            role,
            name: Some(name.into()),
        }
    }
}

impl From<Role> for Author {
    fn from(role: Role) -> Self {
        Self { role, name: None }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
    Developer,
    Tool,
}

impl TryFrom<&str> for Role {
    type Error = &'static str;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "system" => Ok(Role::System),
            "developer" => Ok(Role::Developer),
            "tool" => Ok(Role::Tool),
            _ => Err("Unknown role"),
        }
    }
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Developer => "developer",
            Role::Tool => "tool",
        }
    }
}

impl Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Content {
    Text(TextContent),
    /// Special content for system-level instructions
    SystemContent(SystemContent),
    /// Special content for developer-level instructions
    DeveloperContent(DeveloperContent),
}

impl<T> From<T> for Content
where
    T: Into<String>,
{
    fn from(text: T) -> Self {
        Self::Text(TextContent { text: text.into() })
    }
}

impl From<SystemContent> for Content {
    fn from(sys: SystemContent) -> Self {
        Self::SystemContent(sys)
    }
}

impl From<DeveloperContent> for Content {
    fn from(dev: DeveloperContent) -> Self {
        Self::DeveloperContent(dev)
    }
}

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Message {
    /// An object representing the author of the message, including
    /// their role (e.g., user, assistant) and any additional metadata.
    #[serde(flatten)]
    pub author: Author,

    /// The intended recipient of the message. If not set, the message is
    /// is intend for all (this is the default). Can also be set to specific
    /// identifiers (e.g., 'user', 'assistant', etc.) In the case of a tool call,
    /// the recipient is the name of the tool.
    pub recipient: Option<String>,

    /// The main content of the message. This can be of various types
    /// (e.g., text, code) and structures, depending on the type of `MessageContent` used.
    #[serde(
        deserialize_with = "de_string_or_content_vec",
        serialize_with = "se_string_or_content_vec"
    )]
    pub content: Vec<Content>,

    /// Specifies the target channel (context) for the message, allowing
    /// models to annotate their responses and e.g. control message visibility.
    /// By default, messages do not have channel set (None) and it's not rendered.
    /// When set and render_channel=True, the channel is rendered in message header as
    /// <|channel|>CHANNEL_VALUE (usually, "<|meta_sep|>CHANNEL_VALUE"). To not render
    /// channels, use `formatter.render_channel = False`. (note: parsing will raise an error
    /// if render_channel=False, but a channel was sampled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,

    /// Content type of the message. This is typically only set by the model, you probably don't need to set this.
    pub content_type: Option<String>,
}

impl Message {
    pub fn from_author_and_content<C>(author: Author, content: C) -> Self
    where
        C: Into<Content>,
    {
        Message {
            author,
            content: vec![content.into()],
            channel: None,
            recipient: None,
            content_type: None,
        }
    }

    pub fn from_role_and_content<C>(role: Role, content: C) -> Self
    where
        C: Into<Content>,
    {
        Self::from_author_and_content(Author { role, name: None }, content)
    }

    pub fn from_role_and_contents<I>(role: Role, content: I) -> Self
    where
        I: IntoIterator<Item = Content>,
    {
        Message {
            author: Author { role, name: None },
            content: content.into_iter().collect(),
            channel: None,
            recipient: None,
            content_type: None,
        }
    }
    pub fn adding_content<C>(mut self, content: C) -> Self
    where
        C: Into<Content>,
    {
        self.content.push(content.into());
        self
    }
    pub fn with_channel<S>(mut self, channel: S) -> Self
    where
        S: Into<String>,
    {
        self.channel = Some(channel.into());
        self
    }
    pub fn with_recipient<S>(mut self, recipient: S) -> Self
    where
        S: Into<String>,
    {
        self.recipient = Some(recipient.into());
        self
    }
    pub fn with_content_type<S>(mut self, content_type: S) -> Self
    where
        S: Into<String>,
    {
        self.content_type = Some(content_type.into());
        self
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TextContent {
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct ChannelConfig {
    /// List of valid channels to instruct the model it can generate.
    ///
    /// If empty, this part of the system message will not be rendered.
    pub valid_channels: Vec<String>,

    /// If True, every assistant's message must have channel value set.
    pub channel_required: bool,
}

impl ChannelConfig {
    pub fn require_channels<I, T>(channels: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            valid_channels: channels.into_iter().map(|c| c.into()).collect(),
            channel_required: true,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolNamespaceConfig {
    pub name: String,
    pub description: Option<String>,
    pub tools: Vec<ToolDescription>,
}

impl ToolNamespaceConfig {
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        tools: Vec<ToolDescription>,
    ) -> Self {
        Self {
            name: name.into(),
            description,
            tools,
        }
    }

    pub fn browser() -> Self {
        ToolNamespaceConfig::new(
            "browser",
            Some("Tool for browsing.\nThe `cursor` appears in brackets before each browsing display: `[{cursor}]`.\nCite information from the tool using the following format:\n`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\nDo not quote more than 10 words directly from the tool output.\nsources=web (default: web)".to_string()),
            vec![
                ToolDescription::new(
                    "search",
                    "Searches for information related to `query` and displays `topn` results.",
                    Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "topn": {"type": "number", "default": 10},
                            "source": {"type": "string"}
                        },
                        "required": ["query"]
                    })),
                ),
                ToolDescription::new(
                    "open",
                    "Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\nValid link ids are displayed with the formatting: `【{id}†.*】`.\nIf `cursor` is not provided, the most recent page is implied.\nIf `id` is a string, it is treated as a fully qualified URL associated with `source`.\nIf `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\nUse this function without `id` to scroll to a new location of an opened page.",
                    Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": ["number", "string"],
                                "default": -1
                            },
                            "cursor": {"type": "number", "default": -1},
                            "loc": {"type": "number", "default": -1},
                            "num_lines": {"type": "number", "default": -1},
                            "view_source": {"type": "boolean", "default": false},
                            "source": {"type": "string"}
                        }
                    })),
                ),
                ToolDescription::new(
                    "find",
                    "Finds exact matches of `pattern` in the current page, or the page given by `cursor`.",
                    Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "cursor": {"type": "number", "default": -1}
                        },
                        "required": ["pattern"]
                    })),
                ),
            ],
        )
    }

    pub fn python() -> Self {
        ToolNamespaceConfig::new(
            "python",
            Some("Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n\nWhen you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.".to_string()),
            vec![],
        )
    }
}

/// Content specific to system messages, includes model identity and its instructions
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SystemContent {
    pub model_identity: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub tools: Option<BTreeMap<String, ToolNamespaceConfig>>,
    /// Date/Time at which the conversation is taking place.
    /// Must be an isoformat date for portability to javascript.
    pub conversation_start_date: Option<String>,

    /// The date at which the model's training data ends.
    pub knowledge_cutoff: Option<String>,

    /// Channel configuration for the system message.
    pub channel_config: Option<ChannelConfig>,
}

impl Default for SystemContent {
    fn default() -> Self {
        Self {
            model_identity: Some(
                "You are ChatGPT, a large language model trained by OpenAI.".to_string(),
            ),
            reasoning_effort: Some(ReasoningEffort::Medium),
            tools: None,
            conversation_start_date: None,
            knowledge_cutoff: Some("2024-06".to_string()),
            channel_config: Some(ChannelConfig::require_channels([
                "analysis",
                "commentary",
                "final",
            ])),
        }
    }
}

impl SystemContent {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn with_model_identity(mut self, model_identity: impl Into<String>) -> Self {
        self.model_identity = Some(model_identity.into());
        self
    }
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }
    pub fn with_tools(mut self, ns_config: ToolNamespaceConfig) -> Self {
        let ns = ns_config.name.clone();
        if let Some(ref mut map) = self.tools {
            map.insert(ns, ns_config);
        } else {
            let mut map = BTreeMap::new();
            map.insert(ns, ns_config);
            self.tools = Some(map);
        }
        self
    }
    pub fn with_conversation_start_date(
        mut self,
        conversation_start_date: impl Into<String>,
    ) -> Self {
        self.conversation_start_date = Some(conversation_start_date.into());
        self
    }
    pub fn with_knowledge_cutoff(mut self, knowledge_cutoff: impl Into<String>) -> Self {
        self.knowledge_cutoff = Some(knowledge_cutoff.into());
        self
    }
    pub fn with_channel_config(mut self, channel_config: ChannelConfig) -> Self {
        self.channel_config = Some(channel_config);
        self
    }
    pub fn with_required_channels<I, T>(mut self, channels: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.channel_config = Some(ChannelConfig::require_channels(channels));
        self
    }

    pub fn with_browser_tool(mut self) -> Self {
        self = self.with_tools(ToolNamespaceConfig::browser());
        self
    }

    pub fn with_python_tool(mut self) -> Self {
        self = self.with_tools(ToolNamespaceConfig::python());
        self
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolDescription {
    pub name: String,
    pub description: String,
    pub parameters: Option<serde_json::Value>,
}

impl ToolDescription {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Option<serde_json::Value>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Conversation {
    pub messages: Vec<Message>,
}

impl Conversation {
    pub fn from_messages<I>(messages: I) -> Self
    where
        I: IntoIterator<Item = Message>,
    {
        Self {
            messages: messages.into_iter().collect(),
        }
    }
}

impl<'a> IntoIterator for &'a Conversation {
    type Item = &'a Message;
    type IntoIter = std::slice::Iter<'a, Message>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter()
    }
}

fn de_string_or_content_vec<'de, D>(deserializer: D) -> Result<Vec<Content>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrContentVec(PhantomData<fn() -> Vec<Content>>);

    impl<'de> Visitor<'de> for StringOrContentVec {
        type Value = Vec<Content>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or list of content")
        }

        fn visit_str<E>(self, value: &str) -> Result<Vec<Content>, E>
        where
            E: de::Error,
        {
            Ok(vec![Content::Text(TextContent {
                text: value.to_owned(),
            })])
        }

        fn visit_seq<A>(self, seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(StringOrContentVec(PhantomData))
}

fn se_string_or_content_vec<S>(value: &Vec<Content>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if value.len() == 1 {
        if let Content::Text(TextContent { text }) = &value[0] {
            return serializer.serialize_str(text);
        }
    }
    value.serialize(serializer)
}

/// Content specific to developer messages, includes developer identity and its instructions
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct DeveloperContent {
    pub instructions: Option<String>,
    pub tools: Option<BTreeMap<String, ToolNamespaceConfig>>,
}

impl DeveloperContent {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
    pub fn with_tools(mut self, ns_config: ToolNamespaceConfig) -> Self {
        let ns = ns_config.name.clone();
        if let Some(ref mut map) = self.tools {
            map.insert(ns, ns_config);
        } else {
            let mut map = BTreeMap::new();
            map.insert(ns, ns_config);
            self.tools = Some(map);
        }
        self
    }
    pub fn with_function_tools(mut self, tools: Vec<ToolDescription>) -> Self {
        self = self.with_tools(ToolNamespaceConfig::new("functions", None, tools));
        self
    }
}
