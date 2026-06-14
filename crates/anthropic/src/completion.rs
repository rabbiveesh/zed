use anyhow::Result;
use collections::HashMap;
use futures::{Stream, StreamExt};
use language_model_core::{
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelRequest,
    LanguageModelToolChoice, LanguageModelToolResultContent, LanguageModelToolUse, MessageContent,
    Role, StopReason, TokenUsage,
    util::{fix_streamed_json, parse_tool_arguments},
};
use std::pin::Pin;
use std::str::FromStr;

use crate::{
    AdaptiveThinkingDisplay, AnthropicError, AnthropicModelMode, CacheControl, CacheControlType,
    CacheTtl, ContentDelta, Event, ImageSource, Message, RequestContent, ResponseContent,
    StringOrContents, Thinking, Tool, ToolChoice, ToolResultContent, ToolResultPart, Usage,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnthropicPromptCacheMode {
    Disabled,
    Legacy,
    #[default]
    Automatic,
}

/// How many conversation cache breakpoints to place. Anthropic allows 4 cache
/// breakpoints per request; one is reserved for the long-lived tools/system
/// anchor, leaving three for the conversation — spent on the conversation tail
/// plus the most recent stable turn boundaries.
const CONVERSATION_CACHE_BREAKPOINTS: usize = 3;

fn set_cache_control(content: &mut RequestContent, cache_control: Option<CacheControl>) -> bool {
    match content {
        // Anthropic rejects `cache_control` on (redacted) thinking blocks
        // ("Extra inputs are not permitted"), so these can't hold a breakpoint.
        RequestContent::RedactedThinking { .. } | RequestContent::Thinking { .. } => false,
        RequestContent::Text {
            cache_control: target,
            ..
        }
        | RequestContent::Image {
            cache_control: target,
            ..
        }
        | RequestContent::ToolUse {
            cache_control: target,
            ..
        }
        | RequestContent::ToolResult {
            cache_control: target,
            ..
        } => {
            *target = cache_control;
            true
        }
    }
}

fn mark_last_cacheable_content(content: &mut [RequestContent], cache_control: CacheControl) {
    for content in content.iter_mut().rev() {
        if set_cache_control(content, Some(cache_control)) {
            break;
        }
    }
}

fn to_anthropic_content(content: MessageContent) -> Option<RequestContent> {
    match content {
        MessageContent::Text(text) => {
            let text = if text.chars().last().is_some_and(|c| c.is_whitespace()) {
                text.trim_end().to_string()
            } else {
                text
            };
            if !text.is_empty() {
                Some(RequestContent::Text {
                    text,
                    cache_control: None,
                })
            } else {
                None
            }
        }
        MessageContent::Thinking {
            text: thinking,
            signature,
        } => {
            if let Some(signature) = signature
                && !thinking.is_empty()
            {
                Some(RequestContent::Thinking {
                    thinking,
                    signature,
                    cache_control: None,
                })
            } else {
                None
            }
        }
        MessageContent::RedactedThinking(data) => {
            if !data.is_empty() {
                Some(RequestContent::RedactedThinking { data })
            } else {
                None
            }
        }
        MessageContent::Image(image) => Some(RequestContent::Image {
            source: ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: image.source.to_string(),
            },
            cache_control: None,
        }),
        MessageContent::ToolUse(tool_use) => Some(RequestContent::ToolUse {
            id: tool_use.id.to_string(),
            name: tool_use.name.to_string(),
            input: tool_use.input,
            cache_control: None,
        }),
        MessageContent::ToolResult(tool_result) => {
            let content = match tool_result.content.as_slice() {
                [LanguageModelToolResultContent::Text(text)] => {
                    ToolResultContent::Plain(text.to_string())
                }
                _ => {
                    let parts = tool_result
                        .content
                        .into_iter()
                        .map(|part| match part {
                            LanguageModelToolResultContent::Text(text) => ToolResultPart::Text {
                                text: text.to_string(),
                            },
                            LanguageModelToolResultContent::Image(image) => ToolResultPart::Image {
                                source: ImageSource {
                                    source_type: "base64".to_string(),
                                    media_type: "image/png".to_string(),
                                    data: image.source.to_string(),
                                },
                            },
                        })
                        .collect();
                    ToolResultContent::Multipart(parts)
                }
            };
            Some(RequestContent::ToolResult {
                tool_use_id: tool_result.tool_use_id.to_string(),
                is_error: tool_result.is_error,
                content,
                cache_control: None,
            })
        }
    }
}

pub fn into_anthropic(
    request: LanguageModelRequest,
    model: String,
    default_temperature: f32,
    max_output_tokens: u64,
    mode: AnthropicModelMode,
    cache_mode: AnthropicPromptCacheMode,
) -> crate::Request {
    let mut new_messages: Vec<Message> = Vec::new();
    let mut system_message = String::new();
    let mut any_message_wants_cache = false;

    for message in request.messages {
        if message.contents_empty() {
            continue;
        }

        any_message_wants_cache |= message.cache;

        match message.role {
            Role::User | Role::Assistant => {
                let mut anthropic_message_content: Vec<RequestContent> = message
                    .content
                    .into_iter()
                    .filter_map(to_anthropic_content)
                    .collect();
                let anthropic_role = match message.role {
                    Role::User => crate::Role::User,
                    Role::Assistant => crate::Role::Assistant,
                    Role::System => unreachable!("System role should never occur here"),
                };
                if anthropic_message_content.is_empty() {
                    continue;
                }

                if cache_mode == AnthropicPromptCacheMode::Legacy && message.cache {
                    mark_last_cacheable_content(
                        &mut anthropic_message_content,
                        CacheControl {
                            cache_type: CacheControlType::Ephemeral,
                            ttl: None,
                        },
                    );
                }

                if let Some(last_message) = new_messages.last_mut()
                    && last_message.role == anthropic_role
                {
                    last_message.content.extend(anthropic_message_content);
                    continue;
                }

                new_messages.push(Message {
                    role: anthropic_role,
                    content: anthropic_message_content,
                });
            }
            Role::System => {
                if !system_message.is_empty() {
                    system_message.push_str("\n\n");
                }
                system_message.push_str(&message.string_contents());
            }
        }
    }

    // When caching is enabled, anchor the static prefix (tools + system) with a
    // single explicit long-TTL breakpoint. A breakpoint on the system block
    // caches tools + system together (the prefix renders tools → system →
    // messages), so a separate tool breakpoint is only needed when there is no
    // system prompt. Spending just one breakpoint here frees the rest for the
    // conversation. Anthropic requires longer TTLs to appear earlier in the
    // prefix, and the long-TTL prefix anchor precedes the short-TTL conversation
    // breakpoints below, so the mix is valid.
    let caching_enabled =
        cache_mode == AnthropicPromptCacheMode::Automatic && any_message_wants_cache;
    let long_lived_cache = caching_enabled.then_some(CacheControl {
        cache_type: CacheControlType::Ephemeral,
        ttl: Some(CacheTtl::OneHour),
    });

    let has_system_prompt = !system_message.is_empty();
    let system = if !has_system_prompt {
        None
    } else if let Some(cache_control) = long_lived_cache {
        Some(StringOrContents::Content(vec![RequestContent::Text {
            text: system_message,
            cache_control: Some(cache_control),
        }]))
    } else {
        Some(StringOrContents::String(system_message))
    };

    let mut tools: Vec<Tool> = request
        .tools
        .into_iter()
        .map(|tool| Tool {
            name: tool.name,
            description: tool.description,
            input_schema: tool.input_schema,
            eager_input_streaming: tool.use_input_streaming,
            cache_control: None,
        })
        .collect();
    if let Some(cache_control) = long_lived_cache
        && !has_system_prompt
        && let Some(last_tool) = tools.last_mut()
    {
        last_tool.cache_control = Some(cache_control);
    }

    // Place short-TTL conversation breakpoints on stable turn boundaries plus
    // the conversation tail.
    //
    // A turn boundary — the last block of the assistant/tool message before a
    // real user prompt — is a byte-stable absolute position: later turns are
    // only appended after it, so its prefix never changes and a breakpoint there
    // re-lands on the exact block (and prefix hash) an earlier request wrote.
    // That lets a turn which appends a large batch of parallel tool calls read
    // the whole prior conversation through a boundary instead of rewriting it.
    //
    // We keep the *last few* boundaries, not just the most recent one, because
    // Anthropic's cache has read-after-write latency: an entry written by the
    // immediately-preceding request is not yet readable by the very next request
    // (confirmed empirically — a breakpoint on the exact just-written block with
    // a byte-identical prefix still missed, and only hit a request later). If we
    // anchored solely on the most recent boundary, the turn right after a big
    // batch would find that boundary still "fresh", fall all the way back to the
    // system prefix, and re-create the entire conversation (a partial #58063
    // regression). An older boundary is already committed, so it reliably reads
    // the bulk of the conversation; only the newest, not-yet-committed blocks are
    // re-created, and they read back a turn later (#58063).
    if caching_enabled {
        // Thinking blocks can't hold a breakpoint, so track which absolute
        // positions are markable and snap each target onto the nearest markable
        // block at or before it.
        let markable: Vec<bool> = new_messages
            .iter()
            .flat_map(|message| message.content.iter())
            .map(|block| {
                matches!(
                    block,
                    RequestContent::Text { .. }
                        | RequestContent::Image { .. }
                        | RequestContent::ToolUse { .. }
                        | RequestContent::ToolResult { .. }
                )
            })
            .collect();
        let addresses: Vec<(usize, usize)> = new_messages
            .iter()
            .enumerate()
            .flat_map(|(message_ix, message)| {
                (0..message.content.len()).map(move |block_ix| (message_ix, block_ix))
            })
            .collect();
        let snap = |pos: usize| (0..=pos).rev().find(|&ix| markable[ix]);

        if let Some(last) = addresses.len().checked_sub(1).and_then(|ix| snap(ix)) {
            // First absolute block index of each message.
            let mut message_start = Vec::with_capacity(new_messages.len());
            let mut offset = 0;
            for message in &new_messages {
                message_start.push(offset);
                offset += message.content.len();
            }

            // The tail of the turn before each real user prompt, snapped onto a
            // markable block. These are the stable boundaries we anchor on.
            let mut boundaries: Vec<usize> = new_messages
                .iter()
                .enumerate()
                .filter_map(|(message_ix, message)| {
                    let is_user_prompt = message.role == crate::Role::User
                        && message
                            .content
                            .iter()
                            .any(|block| matches!(block, RequestContent::Text { .. }));
                    let turn_start = (is_user_prompt && message_ix > 0).then_some(message_ix)?;
                    let previous = turn_start - 1;
                    let previous_tail = message_start[previous]
                        + new_messages[previous].content.len().checked_sub(1)?;
                    snap(previous_tail)
                })
                .collect();
            boundaries.dedup();

            // Reserve one breakpoint for the conversation tail (so the newest
            // blocks are written for a later turn to read) and spend the rest on
            // the most recent boundaries.
            let kept_boundaries = CONVERSATION_CACHE_BREAKPOINTS.saturating_sub(1);
            let mut breakpoints: Vec<usize> = boundaries
                .into_iter()
                .rev()
                .take(kept_boundaries)
                .chain(std::iter::once(last))
                .collect();
            breakpoints.sort_unstable();
            breakpoints.dedup();

            for position in breakpoints {
                let (message_ix, block_ix) = addresses[position];
                set_cache_control(
                    &mut new_messages[message_ix].content[block_ix],
                    Some(CacheControl {
                        cache_type: CacheControlType::Ephemeral,
                        ttl: None,
                    }),
                );
            }
        }
    }

    crate::Request {
        model,
        messages: new_messages,
        max_tokens: max_output_tokens,
        system,
        // Conversation caching is handled by the explicit rolling breakpoints
        // placed on the messages above, so the top-level automatic breakpoint is
        // left unset to stay within Anthropic's 4-breakpoint budget.
        cache_control: None,
        thinking: if request.thinking_allowed {
            match mode {
                AnthropicModelMode::Thinking { budget_tokens } => {
                    Some(Thinking::Enabled { budget_tokens })
                }
                AnthropicModelMode::AdaptiveThinking => Some(Thinking::Adaptive {
                    display: Some(AdaptiveThinkingDisplay::Summarized),
                }),
                AnthropicModelMode::Default => None,
            }
        } else {
            None
        },
        tools,
        tool_choice: request.tool_choice.map(|choice| match choice {
            LanguageModelToolChoice::Auto => ToolChoice::Auto,
            LanguageModelToolChoice::Any => ToolChoice::Any,
            LanguageModelToolChoice::None => ToolChoice::None,
        }),
        metadata: None,
        output_config: if request.thinking_allowed
            && matches!(mode, AnthropicModelMode::AdaptiveThinking)
        {
            request.thinking_effort.as_deref().and_then(|effort| {
                let effort = match effort {
                    "low" => Some(crate::Effort::Low),
                    "medium" => Some(crate::Effort::Medium),
                    "high" => Some(crate::Effort::High),
                    "xhigh" => Some(crate::Effort::XHigh),
                    "max" => Some(crate::Effort::Max),
                    _ => None,
                };
                effort.map(|effort| crate::OutputConfig {
                    effort: Some(effort),
                })
            })
        } else {
            None
        },
        stop_sequences: Vec::new(),
        speed: request.speed.map(Into::into),
        temperature: request.temperature.or(Some(default_temperature)),
        top_k: None,
        top_p: None,
    }
}

pub struct AnthropicEventMapper {
    tool_uses_by_index: HashMap<usize, RawToolUse>,
    usage: Usage,
    stop_reason: StopReason,
}

impl AnthropicEventMapper {
    pub fn new() -> Self {
        Self {
            tool_uses_by_index: HashMap::default(),
            usage: Usage::default(),
            stop_reason: StopReason::EndTurn,
        }
    }

    pub fn map_stream(
        mut self,
        events: Pin<Box<dyn Send + Stream<Item = Result<Event, AnthropicError>>>>,
    ) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        events.flat_map(move |event| {
            futures::stream::iter(match event {
                Ok(event) => self.map_event(event),
                Err(error) => vec![Err(error.into())],
            })
        })
    }

    pub fn map_event(
        &mut self,
        event: Event,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        match event {
            Event::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                ResponseContent::Text { text } => {
                    vec![Ok(LanguageModelCompletionEvent::Text(text))]
                }
                ResponseContent::Thinking { thinking } => {
                    vec![Ok(LanguageModelCompletionEvent::Thinking {
                        text: thinking,
                        signature: None,
                    })]
                }
                ResponseContent::RedactedThinking { data } => {
                    vec![Ok(LanguageModelCompletionEvent::RedactedThinking { data })]
                }
                ResponseContent::ToolUse { id, name, .. } => {
                    self.tool_uses_by_index.insert(
                        index,
                        RawToolUse {
                            id,
                            name,
                            input_json: String::new(),
                        },
                    );
                    Vec::new()
                }
            },
            Event::ContentBlockDelta { index, delta } => match delta {
                ContentDelta::TextDelta { text } => {
                    vec![Ok(LanguageModelCompletionEvent::Text(text))]
                }
                ContentDelta::ThinkingDelta { thinking } => {
                    vec![Ok(LanguageModelCompletionEvent::Thinking {
                        text: thinking,
                        signature: None,
                    })]
                }
                ContentDelta::SignatureDelta { signature } => {
                    vec![Ok(LanguageModelCompletionEvent::Thinking {
                        text: "".to_string(),
                        signature: Some(signature),
                    })]
                }
                ContentDelta::InputJsonDelta { partial_json } => {
                    if let Some(tool_use) = self.tool_uses_by_index.get_mut(&index) {
                        tool_use.input_json.push_str(&partial_json);

                        // Try to convert invalid (incomplete) JSON into
                        // valid JSON that serde can accept, e.g. by closing
                        // unclosed delimiters. This way, we can update the
                        // UI with whatever has been streamed back so far.
                        if let Ok(input) =
                            serde_json::Value::from_str(&fix_streamed_json(&tool_use.input_json))
                        {
                            return vec![Ok(LanguageModelCompletionEvent::ToolUse(
                                LanguageModelToolUse {
                                    id: tool_use.id.clone().into(),
                                    name: tool_use.name.clone().into(),
                                    is_input_complete: false,
                                    raw_input: tool_use.input_json.clone(),
                                    input,
                                    thought_signature: None,
                                },
                            ))];
                        }
                    }
                    vec![]
                }
            },
            Event::ContentBlockStop { index } => {
                if let Some(tool_use) = self.tool_uses_by_index.remove(&index) {
                    let input_json = tool_use.input_json.trim();
                    let event_result = match parse_tool_arguments(input_json) {
                        Ok(input) => Ok(LanguageModelCompletionEvent::ToolUse(
                            LanguageModelToolUse {
                                id: tool_use.id.into(),
                                name: tool_use.name.into(),
                                is_input_complete: true,
                                input,
                                raw_input: tool_use.input_json.clone(),
                                thought_signature: None,
                            },
                        )),
                        Err(json_parse_err) => {
                            Ok(LanguageModelCompletionEvent::ToolUseJsonParseError {
                                id: tool_use.id.into(),
                                tool_name: tool_use.name.into(),
                                raw_input: input_json.into(),
                                json_parse_error: json_parse_err.to_string(),
                            })
                        }
                    };

                    vec![event_result]
                } else {
                    Vec::new()
                }
            }
            Event::MessageStart { message } => {
                update_usage(&mut self.usage, &message.usage);
                vec![
                    Ok(LanguageModelCompletionEvent::UsageUpdate(convert_usage(
                        &self.usage,
                    ))),
                    Ok(LanguageModelCompletionEvent::StartMessage {
                        message_id: message.id,
                    }),
                ]
            }
            Event::MessageDelta { delta, usage } => {
                update_usage(&mut self.usage, &usage);
                if let Some(stop_reason) = delta.stop_reason.as_deref() {
                    self.stop_reason = match stop_reason {
                        "end_turn" => StopReason::EndTurn,
                        "max_tokens" => StopReason::MaxTokens,
                        "tool_use" => StopReason::ToolUse,
                        "refusal" => StopReason::Refusal,
                        _ => {
                            log::error!("Unexpected anthropic stop_reason: {stop_reason}");
                            StopReason::EndTurn
                        }
                    };
                }
                vec![Ok(LanguageModelCompletionEvent::UsageUpdate(
                    convert_usage(&self.usage),
                ))]
            }
            Event::MessageStop => {
                vec![Ok(LanguageModelCompletionEvent::Stop(self.stop_reason))]
            }
            Event::Error { error } => {
                vec![Err(error.into())]
            }
            _ => Vec::new(),
        }
    }
}

struct RawToolUse {
    id: String,
    name: String,
    input_json: String,
}

/// Updates usage data by preferring counts from `new`.
fn update_usage(usage: &mut Usage, new: &Usage) {
    if let Some(input_tokens) = new.input_tokens {
        usage.input_tokens = Some(input_tokens);
    }
    if let Some(output_tokens) = new.output_tokens {
        usage.output_tokens = Some(output_tokens);
    }
    if let Some(cache_creation_input_tokens) = new.cache_creation_input_tokens {
        usage.cache_creation_input_tokens = Some(cache_creation_input_tokens);
    }
    if let Some(cache_read_input_tokens) = new.cache_read_input_tokens {
        usage.cache_read_input_tokens = Some(cache_read_input_tokens);
    }
}

fn convert_usage(usage: &Usage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens.unwrap_or(0),
        output_tokens: usage.output_tokens.unwrap_or(0),
        cache_creation_input_tokens: usage.cache_creation_input_tokens.unwrap_or(0),
        cache_read_input_tokens: usage.cache_read_input_tokens.unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnthropicModelMode;
    use language_model_core::{LanguageModelImage, LanguageModelRequestMessage, MessageContent};

    #[test]
    fn test_caching_anchors_system_and_rolls_conversation_breakpoints() {
        let request = LanguageModelRequest {
            messages: vec![
                LanguageModelRequestMessage {
                    role: Role::System,
                    content: vec![MessageContent::Text("You are helpful.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![
                        MessageContent::Text("Some prompt".to_string()),
                        MessageContent::Image(LanguageModelImage::empty()),
                        MessageContent::Image(LanguageModelImage::empty()),
                    ],
                    cache: true,
                    reasoning_details: None,
                },
            ],
            thread_id: None,
            prompt_id: None,
            intent: None,
            stop: vec![],
            temperature: None,
            tools: vec![language_model_core::LanguageModelRequestTool {
                name: "do_thing".into(),
                description: "Does a thing.".into(),
                input_schema: serde_json::json!({"type": "object"}),
                use_input_streaming: false,
            }],
            tool_choice: None,
            thinking_allowed: true,
            thinking_effort: None,
            speed: None,
        };

        let anthropic_request = into_anthropic(
            request,
            "claude-3-5-sonnet".to_string(),
            0.7,
            4096,
            AnthropicModelMode::Default,
            AnthropicPromptCacheMode::Automatic,
        );

        // The conversation tail carries an explicit short-TTL breakpoint, and
        // every conversation breakpoint is a short-TTL ephemeral marker.
        assert_eq!(anthropic_request.messages.len(), 1);
        let content = &anthropic_request.messages[0].content;
        let cache_control_of = |block: &RequestContent| match block {
            RequestContent::Text { cache_control, .. }
            | RequestContent::Thinking { cache_control, .. }
            | RequestContent::Image { cache_control, .. }
            | RequestContent::ToolUse { cache_control, .. }
            | RequestContent::ToolResult { cache_control, .. } => *cache_control,
            RequestContent::RedactedThinking { .. } => None,
        };
        assert!(
            matches!(
                cache_control_of(content.last().expect("message has content")),
                Some(CacheControl {
                    cache_type: CacheControlType::Ephemeral,
                    ttl: None,
                })
            ),
            "the conversation tail should carry a short-TTL breakpoint",
        );
        let marked = content.iter().filter(|b| cache_control_of(b).is_some()).count();
        assert!(
            marked >= 1 && marked <= CONVERSATION_CACHE_BREAKPOINTS,
            "expected up to {CONVERSATION_CACHE_BREAKPOINTS} conversation breakpoints, got {marked}",
        );

        // The top-level automatic breakpoint is no longer used; conversation
        // caching is handled by the explicit breakpoints above.
        assert!(anthropic_request.cache_control.is_none());

        // System prompt is emitted in array form with a long-TTL breakpoint on
        // the final text block.
        match anthropic_request.system {
            Some(StringOrContents::Content(ref blocks)) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(
                    blocks[0],
                    RequestContent::Text {
                        cache_control: Some(CacheControl {
                            cache_type: CacheControlType::Ephemeral,
                            ttl: Some(CacheTtl::OneHour),
                        }),
                        ..
                    }
                ));
            }
            other => panic!("expected system content array, got {other:?}"),
        }

        // With a system prompt present, the system breakpoint already caches
        // tools + system together, so no separate tool breakpoint is spent.
        assert_eq!(anthropic_request.tools.len(), 1);
        assert!(anthropic_request.tools[0].cache_control.is_none());
    }

    /// Helpers shared by the breakpoint-placement tests.
    fn block_is_marked(block: &RequestContent) -> bool {
        matches!(
            block,
            RequestContent::Text { cache_control: Some(_), .. }
                | RequestContent::Thinking { cache_control: Some(_), .. }
                | RequestContent::Image { cache_control: Some(_), .. }
                | RequestContent::ToolUse { cache_control: Some(_), .. }
                | RequestContent::ToolResult { cache_control: Some(_), .. }
        )
    }

    fn marked_block_indices(request: &crate::Request) -> Vec<usize> {
        request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .enumerate()
            .filter_map(|(ix, block)| block_is_marked(block).then_some(ix))
            .collect()
    }

    fn parallel_batch(
        text: &str,
        count: usize,
    ) -> (Vec<MessageContent>, Vec<MessageContent>) {
        use language_model_core::{
            LanguageModelToolResult, LanguageModelToolResultContent, LanguageModelToolUse,
        };
        let mut assistant = vec![MessageContent::Text(text.to_string())];
        let mut results = Vec::new();
        for i in 0..count {
            let id = format!("tool_{i}");
            assistant.push(MessageContent::ToolUse(LanguageModelToolUse {
                id: id.clone().into(),
                name: "read_file".into(),
                raw_input: "{}".to_string(),
                input: serde_json::json!({ "path": format!("src/file_{i}.rs") }),
                is_input_complete: true,
                thought_signature: None,
            }));
            results.push(MessageContent::ToolResult(LanguageModelToolResult {
                tool_use_id: id.into(),
                tool_name: "read_file".into(),
                is_error: false,
                content: vec![LanguageModelToolResultContent::Text("…file contents…".into())],
                output: None,
            }));
        }
        (assistant, results)
    }

    fn user_message(text: &str) -> LanguageModelRequestMessage {
        LanguageModelRequestMessage {
            role: Role::User,
            content: vec![MessageContent::Text(text.to_string())],
            cache: false,
            reasoning_details: None,
        }
    }

    fn assistant_message(content: Vec<MessageContent>) -> LanguageModelRequestMessage {
        LanguageModelRequestMessage {
            role: Role::Assistant,
            content,
            cache: false,
            reasoning_details: None,
        }
    }

    fn caching_request(messages: Vec<LanguageModelRequestMessage>) -> crate::Request {
        let mut messages = messages;
        if let Some(last) = messages.last_mut() {
            last.cache = true;
        }
        let request = LanguageModelRequest {
            messages,
            thread_id: None,
            prompt_id: None,
            intent: None,
            stop: vec![],
            temperature: None,
            tools: vec![language_model_core::LanguageModelRequestTool {
                name: "read_file".into(),
                description: "Reads a file.".into(),
                input_schema: serde_json::json!({"type": "object"}),
                use_input_streaming: false,
            }],
            tool_choice: None,
            thinking_allowed: true,
            thinking_effort: None,
            speed: None,
        };
        into_anthropic(
            request,
            "claude-3-5-sonnet".to_string(),
            0.7,
            4096,
            AnthropicModelMode::Default,
            AnthropicPromptCacheMode::Automatic,
        )
    }

    /// Fix for #58063: a turn that appends a big batch of parallel tool calls —
    /// far more blocks than Anthropic's cache lookback window — still anchors on
    /// the previous turn's stable boundary. A breakpoint on that boundary reads
    /// the whole prior conversation regardless of the batch size, so only the new
    /// blocks are written instead of the entire conversation being re-created.
    #[test]
    fn test_big_batch_turn_anchors_on_prior_boundary_and_tail() {
        let (batch, results) = parallel_batch("Reading the files…", 30);
        let anthropic_request = caching_request(vec![
            LanguageModelRequestMessage {
                role: Role::System,
                content: vec![MessageContent::Text("You are helpful.".to_string())],
                cache: false,
                reasoning_details: None,
            },
            user_message("First question."),
            assistant_message(vec![MessageContent::Text(
                "Here is turn one's answer.".to_string(),
            )]),
            // Current turn: a new prompt, then a large parallel batch.
            user_message("Now read a bunch of files."),
            assistant_message(batch),
            LanguageModelRequestMessage {
                role: Role::User,
                content: results,
                cache: true,
                reasoning_details: None,
            },
        ]);

        let flat: Vec<&RequestContent> = anthropic_request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .collect();
        let breakpoints = marked_block_indices(&anthropic_request);

        // The conversation tail is always marked so the new blocks are written.
        assert_eq!(breakpoints.last(), Some(&(flat.len() - 1)));

        // The previous turn's boundary (the tail of "turn one's answer") is also
        // marked, across a gap far larger than the lookback window — the boundary
        // is read by exact position, no chaining required.
        let boundary_ix = flat
            .iter()
            .position(|block| {
                matches!(block, RequestContent::Text { text, .. } if text.contains("turn one's answer"))
            })
            .expect("previous turn boundary present");
        assert!(
            breakpoints.contains(&boundary_ix),
            "previous turn's boundary should be anchored (breakpoints: {breakpoints:?}, boundary: {boundary_ix})",
        );
        assert!(
            breakpoints.len() <= CONVERSATION_CACHE_BREAKPOINTS,
            "at most {CONVERSATION_CACHE_BREAKPOINTS} conversation breakpoints, got {breakpoints:?}",
        );
        assert!(anthropic_request.cache_control.is_none());
    }

    /// Fix for #58063 (the read-after-write case): a cache entry written by the
    /// immediately-preceding request is not yet readable on the next request, so
    /// anchoring on only the most recent boundary would make the turn right after
    /// a big batch fall back to the system prefix and re-create the whole
    /// conversation. We keep the last *several* boundaries, so an older, already
    /// committed one is always available to read the bulk of the conversation.
    #[test]
    fn test_keeps_multiple_turn_boundaries() {
        let (batch, results) = parallel_batch("Reading now…", 30);
        let anthropic_request = caching_request(vec![
            LanguageModelRequestMessage {
                role: Role::System,
                content: vec![MessageContent::Text("You are helpful.".to_string())],
                cache: false,
                reasoning_details: None,
            },
            user_message("First question."),
            assistant_message(vec![MessageContent::Text("Turn one answer.".to_string())]),
            user_message("Second question."),
            assistant_message(vec![MessageContent::Text("Turn two answer.".to_string())]),
            user_message("Now read a bunch of files."),
            assistant_message(batch),
            LanguageModelRequestMessage {
                role: Role::User,
                content: results,
                cache: true,
                reasoning_details: None,
            },
        ]);

        let flat: Vec<&RequestContent> = anthropic_request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .collect();
        let breakpoints = marked_block_indices(&anthropic_request);

        let boundary_of = |needle: &str| {
            flat.iter()
                .position(|block| {
                    matches!(block, RequestContent::Text { text, .. } if text.contains(needle))
                })
                .unwrap_or_else(|| panic!("boundary {needle:?} present"))
        };
        // Both recent boundaries are anchored — the most recent (turn two) and the
        // older, definitely-committed one (turn one) — plus the tail.
        let recent_boundary = boundary_of("Turn two answer.");
        let older_boundary = boundary_of("Turn one answer.");
        assert!(
            breakpoints.contains(&older_boundary),
            "an older committed boundary must be kept for read-after-write fallback (breakpoints: {breakpoints:?})",
        );
        assert!(
            breakpoints.contains(&recent_boundary),
            "the most recent boundary should be anchored (breakpoints: {breakpoints:?})",
        );
        assert_eq!(breakpoints.last(), Some(&(flat.len() - 1)));
        assert_eq!(
            breakpoints.len(),
            CONVERSATION_CACHE_BREAKPOINTS,
            "expected tail + two boundaries, got {breakpoints:?}",
        );
    }

    /// Regression test: Anthropic rejects `cache_control` on thinking blocks, so
    /// breakpoints must never land on one. Verifies no thinking block is marked
    /// even when thinking blocks sit on the boundaries we anchor.
    #[test]
    fn test_breakpoints_never_land_on_thinking_blocks() {
        use language_model_core::{
            LanguageModelToolResult, LanguageModelToolResultContent, LanguageModelToolUse,
        };

        let thinking = || MessageContent::Thinking {
            text: "deliberating…".to_string(),
            signature: Some("sig".to_string()),
        };

        const PARALLEL_TOOL_CALLS: usize = 12;
        let mut batch_content = vec![thinking(), MessageContent::Text("Working…".to_string())];
        let mut tool_result_content = Vec::new();
        for i in 0..PARALLEL_TOOL_CALLS {
            let id = format!("tool_{i}");
            batch_content.push(MessageContent::ToolUse(LanguageModelToolUse {
                id: id.clone().into(),
                name: "read_file".into(),
                raw_input: "{}".to_string(),
                input: serde_json::json!({ "path": format!("src/file_{i}.rs") }),
                is_input_complete: true,
                thought_signature: None,
            }));
            tool_result_content.push(MessageContent::ToolResult(LanguageModelToolResult {
                tool_use_id: id.into(),
                tool_name: "read_file".into(),
                is_error: false,
                content: vec![LanguageModelToolResultContent::Text("…contents…".into())],
                output: None,
            }));
        }

        let request = LanguageModelRequest {
            messages: vec![
                LanguageModelRequestMessage {
                    role: Role::System,
                    content: vec![MessageContent::Text("You are helpful.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text("First question.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                LanguageModelRequestMessage {
                    role: Role::Assistant,
                    content: vec![thinking(), MessageContent::Text("Turn one answer.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text("Now read the files.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                LanguageModelRequestMessage {
                    role: Role::Assistant,
                    content: batch_content,
                    cache: false,
                    reasoning_details: None,
                },
                LanguageModelRequestMessage {
                    role: Role::User,
                    content: tool_result_content,
                    cache: true,
                    reasoning_details: None,
                },
            ],
            thread_id: None,
            prompt_id: None,
            intent: None,
            stop: vec![],
            temperature: None,
            tools: vec![language_model_core::LanguageModelRequestTool {
                name: "read_file".into(),
                description: "Reads a file.".into(),
                input_schema: serde_json::json!({"type": "object"}),
                use_input_streaming: false,
            }],
            tool_choice: None,
            thinking_allowed: true,
            thinking_effort: None,
            speed: None,
        };

        let anthropic_request = into_anthropic(
            request,
            "claude-3-5-sonnet".to_string(),
            0.7,
            4096,
            AnthropicModelMode::Default,
            AnthropicPromptCacheMode::Automatic,
        );

        // The request must actually contain thinking blocks, and none may carry
        // a cache_control field.
        let mut saw_thinking = false;
        let mut absolute_ix = 0usize;
        let mut breakpoints = Vec::new();
        for message in &anthropic_request.messages {
            for block in &message.content {
                match block {
                    RequestContent::Thinking { cache_control, .. } => {
                        saw_thinking = true;
                        assert!(
                            cache_control.is_none(),
                            "cache_control must never be set on a thinking block",
                        );
                    }
                    RequestContent::Text { cache_control: Some(_), .. }
                    | RequestContent::Image { cache_control: Some(_), .. }
                    | RequestContent::ToolUse { cache_control: Some(_), .. }
                    | RequestContent::ToolResult { cache_control: Some(_), .. } => {
                        breakpoints.push(absolute_ix);
                    }
                    _ => {}
                }
                absolute_ix += 1;
            }
        }
        assert!(saw_thinking, "test should exercise thinking blocks");

        // At least one breakpoint was placed (tail), and the conversation tail is
        // marked — the thinking blocks on the boundaries were snapped past.
        assert!(
            !breakpoints.is_empty(),
            "expected breakpoints to be placed",
        );
    }

    #[test]
    fn test_legacy_caching_marks_last_message_content_block() {
        let request = LanguageModelRequest {
            messages: vec![
                LanguageModelRequestMessage {
                    role: Role::System,
                    content: vec![MessageContent::Text("You are helpful.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![
                        MessageContent::Text("Some prompt".to_string()),
                        MessageContent::Image(LanguageModelImage::empty()),
                    ],
                    cache: true,
                    reasoning_details: None,
                },
            ],
            thread_id: None,
            prompt_id: None,
            intent: None,
            stop: vec![],
            temperature: None,
            tools: vec![language_model_core::LanguageModelRequestTool {
                name: "do_thing".into(),
                description: "Does a thing.".into(),
                input_schema: serde_json::json!({"type": "object"}),
                use_input_streaming: false,
            }],
            tool_choice: None,
            thinking_allowed: true,
            thinking_effort: None,
            speed: None,
        };

        let anthropic_request = into_anthropic(
            request,
            "claude-3-5-sonnet".to_string(),
            0.7,
            4096,
            AnthropicModelMode::Default,
            AnthropicPromptCacheMode::Legacy,
        );

        assert!(anthropic_request.cache_control.is_none());
        assert!(matches!(
            anthropic_request.system,
            Some(StringOrContents::String(_))
        ));
        assert_eq!(anthropic_request.tools.len(), 1);
        assert!(anthropic_request.tools[0].cache_control.is_none());
        assert_eq!(anthropic_request.messages.len(), 1);
        assert!(matches!(
            anthropic_request.messages[0].content[0],
            RequestContent::Text {
                cache_control: None,
                ..
            }
        ));
        assert!(matches!(
            anthropic_request.messages[0].content[1],
            RequestContent::Image {
                cache_control: Some(CacheControl {
                    cache_type: CacheControlType::Ephemeral,
                    ttl: None,
                }),
                ..
            }
        ));
    }

    #[test]
    fn test_xhigh_effort_is_serialized_for_adaptive_thinking() {
        let request = LanguageModelRequest {
            messages: vec![LanguageModelRequestMessage {
                role: Role::User,
                content: vec![MessageContent::Text("Hi".to_string())],
                cache: false,
                reasoning_details: None,
            }],
            thread_id: None,
            prompt_id: None,
            intent: None,
            stop: vec![],
            temperature: None,
            tools: vec![],
            tool_choice: None,
            thinking_allowed: true,
            thinking_effort: Some("xhigh".into()),
            speed: None,
        };

        let anthropic_request = into_anthropic(
            request,
            "claude-opus-4-8".to_string(),
            1.0,
            128_000,
            AnthropicModelMode::AdaptiveThinking,
            AnthropicPromptCacheMode::Automatic,
        );

        assert_eq!(
            anthropic_request
                .output_config
                .and_then(|config| config.effort),
            Some(crate::Effort::XHigh)
        );
    }

    #[test]
    fn test_no_cache_control_when_caching_disabled() {
        let request = LanguageModelRequest {
            messages: vec![
                LanguageModelRequestMessage {
                    role: Role::System,
                    content: vec![MessageContent::Text("You are helpful.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text("Hi".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
            ],
            thread_id: None,
            prompt_id: None,
            intent: None,
            stop: vec![],
            temperature: None,
            tools: vec![language_model_core::LanguageModelRequestTool {
                name: "do_thing".into(),
                description: "Does a thing.".into(),
                input_schema: serde_json::json!({"type": "object"}),
                use_input_streaming: false,
            }],
            tool_choice: None,
            thinking_allowed: true,
            thinking_effort: None,
            speed: None,
        };

        let anthropic_request = into_anthropic(
            request,
            "claude-3-5-sonnet".to_string(),
            0.7,
            4096,
            AnthropicModelMode::Default,
            AnthropicPromptCacheMode::Automatic,
        );

        assert!(anthropic_request.cache_control.is_none());
        assert!(matches!(
            anthropic_request.system,
            Some(StringOrContents::String(_))
        ));
        assert!(anthropic_request.tools[0].cache_control.is_none());
    }

    fn request_with_assistant_content(assistant_content: Vec<MessageContent>) -> crate::Request {
        let mut request = LanguageModelRequest {
            messages: vec![LanguageModelRequestMessage {
                role: Role::User,
                content: vec![MessageContent::Text("Hello".to_string())],
                cache: false,
                reasoning_details: None,
            }],
            thinking_effort: None,
            thread_id: None,
            prompt_id: None,
            intent: None,
            stop: vec![],
            temperature: None,
            tools: vec![],
            tool_choice: None,
            thinking_allowed: true,
            speed: None,
        };
        request.messages.push(LanguageModelRequestMessage {
            role: Role::Assistant,
            content: assistant_content,
            cache: false,
            reasoning_details: None,
        });
        into_anthropic(
            request,
            "claude-sonnet-4-5".to_string(),
            1.0,
            16000,
            AnthropicModelMode::Thinking {
                budget_tokens: Some(10000),
            },
            AnthropicPromptCacheMode::Automatic,
        )
    }

    #[test]
    fn test_unsigned_thinking_blocks_stripped() {
        let result = request_with_assistant_content(vec![
            MessageContent::Thinking {
                text: "Cancelled mid-think, no signature".to_string(),
                signature: None,
            },
            MessageContent::Text("Some response text".to_string()),
        ]);

        let assistant_message = result
            .messages
            .iter()
            .find(|m| m.role == crate::Role::Assistant)
            .expect("assistant message should still exist");

        assert_eq!(
            assistant_message.content.len(),
            1,
            "Only the text content should remain; unsigned thinking block should be stripped"
        );
        assert!(matches!(
            &assistant_message.content[0],
            RequestContent::Text { text, .. } if text == "Some response text"
        ));
    }

    #[test]
    fn test_signed_thinking_blocks_preserved() {
        let result = request_with_assistant_content(vec![
            MessageContent::Thinking {
                text: "Completed thinking".to_string(),
                signature: Some("valid-signature".to_string()),
            },
            MessageContent::Text("Response".to_string()),
        ]);

        let assistant_message = result
            .messages
            .iter()
            .find(|m| m.role == crate::Role::Assistant)
            .expect("assistant message should exist");

        assert_eq!(
            assistant_message.content.len(),
            2,
            "Both the signed thinking block and text should be preserved"
        );
        assert!(matches!(
            &assistant_message.content[0],
            RequestContent::Thinking { thinking, signature, .. }
                if thinking == "Completed thinking" && signature == "valid-signature"
        ));
    }

    #[test]
    fn test_only_unsigned_thinking_block_omits_entire_message() {
        let result = request_with_assistant_content(vec![MessageContent::Thinking {
            text: "Cancelled before any text or signature".to_string(),
            signature: None,
        }]);

        let assistant_messages: Vec<_> = result
            .messages
            .iter()
            .filter(|m| m.role == crate::Role::Assistant)
            .collect();

        assert_eq!(
            assistant_messages.len(),
            0,
            "An assistant message whose only content was an unsigned thinking block \
             should be omitted entirely"
        );
    }
}
