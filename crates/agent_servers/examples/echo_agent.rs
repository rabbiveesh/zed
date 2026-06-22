//! `echo_agent` — a minimal external ACP agent that streams the user's prompt
//! back as the assistant's response.
//!
//! It exists to reproduce agent-panel performance issues (e.g. zed-industries
//! issue #57349) deterministically and offline: it exercises the exact external
//! ACP path (subprocess + newline-delimited JSON-RPC over stdio + streamed
//! `session/update` chunks) that real agents use, so the panel renders a real,
//! growing, streamed markdown transcript with the generating spinner running —
//! without needing a live model.
//!
//! ## Build
//! ```sh
//! cargo build -p agent_servers --example echo_agent
//! # binary at: target/debug/examples/echo_agent
//! ```
//!
//! ## Register it (settings.json)
//! ```json
//! {
//!   "agent_servers": {
//!     "echo": {
//!       "type": "custom",
//!       "command": "/absolute/path/to/zed/target/debug/examples/echo_agent",
//!       "args": [],
//!       "env": {}
//!     }
//!   }
//! }
//! ```
//! Then pick "echo" as the agent in the panel and send a (long, markdown-heavy)
//! message. Inspect the wire with `dev::OpenAcpLogs`.
//!
//! ## Stress knobs (env vars)
//! - `ECHO_DELAY_MS`   — delay between streamed chunks (default 20). Lower =
//!   higher frame pressure.
//! - `ECHO_CHUNK_CHARS`— characters per streamed chunk (default 6). Smaller =
//!   more chunks = more re-layouts.
//! - `ECHO_REPEAT`     — repeat the echoed text N times (default 1) to inflate
//!   the transcript and amplify per-frame markdown re-shaping.
//! - `ECHO_LINES`      — if > 0, ignore the prompt and stream an N-line
//!   synthetic markdown document instead (e.g. `ECHO_LINES=7000` to mint a
//!   7k-line message like the one in #57349). Default 0 (echo the prompt).
//! - `ECHO_PLAIN`      — with `ECHO_LINES`, emit one giant plain-text paragraph
//!   (≈1 markdown block) instead of structured markdown. Control test: isolates
//!   per-block taffy layout cost from raw message length. Default 0.

use std::io::{BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};

fn main() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Value>();

    // Read JSON-RPC messages off stdin on a dedicated thread. `session/cancel`
    // is observed here directly (setting the shared flag) rather than forwarded,
    // because the main thread is busy sleeping between chunks while streaming a
    // turn and would not otherwise see the cancellation until the turn finished.
    thread::spawn({
        let cancelled = cancelled.clone();
        move || {
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                let Ok(message) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                if message.get("method").and_then(Value::as_str) == Some("session/cancel") {
                    cancelled.store(true, Ordering::SeqCst);
                    continue;
                }
                if tx.send(message).is_err() {
                    break;
                }
            }
            // stdin closed: the client is gone, so tear the whole process down.
            std::process::exit(0);
        }
    });

    let delay = Duration::from_millis(env_u64("ECHO_DELAY_MS", 20));
    let chunk_chars = env_u64("ECHO_CHUNK_CHARS", 6).max(1) as usize;
    let repeat = env_u64("ECHO_REPEAT", 1).max(1) as usize;
    let echo_lines = env_u64("ECHO_LINES", 0) as usize;
    let echo_plain = env_u64("ECHO_PLAIN", 0) != 0;
    let mut session_counter: u64 = 0;

    for message in rx {
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            continue;
        };
        let id = message.get("id").cloned();

        match method {
            "initialize" => respond(
                id,
                json!({ "protocolVersion": 1, "agentCapabilities": {} }),
            ),
            "authenticate" => respond(id, json!({})),
            "session/new" => {
                session_counter += 1;
                respond(id, json!({ "sessionId": format!("echo-{session_counter}") }));
            }
            "session/prompt" => {
                let session_id = message
                    .pointer("/params/sessionId")
                    .cloned()
                    .unwrap_or(Value::Null);
                cancelled.store(false, Ordering::SeqCst);
                // `ECHO_LINES=N` mints a deterministic N-line markdown document
                // (headings, prose, code fences, lists) so a large single
                // message can be reproduced without pasting one. Otherwise echo
                // the prompt text, repeated `ECHO_REPEAT` times.
                let body = if echo_lines > 0 {
                    if echo_plain {
                        generate_plain(echo_lines)
                    } else {
                        generate_markdown(echo_lines)
                    }
                } else {
                    let text = extract_prompt_text(&message);
                    let mut body = String::new();
                    for index in 0..repeat {
                        if index > 0 {
                            body.push_str("\n\n");
                        }
                        body.push_str(&text);
                    }
                    if body.is_empty() {
                        body.push_str("_(echo: empty prompt)_");
                    }
                    body
                };
                let stop_reason = stream_body(&session_id, &body, chunk_chars, delay, &cancelled);
                respond(id, json!({ "stopReason": stop_reason }));
            }
            // Any other request: reply method-not-found so the client moves on
            // instead of waiting for a response that never comes.
            _ => {
                if let Some(id) = id {
                    send(json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": "method not found" },
                    }));
                }
            }
        }
    }
}

/// Stream `body` back as `agent_message_chunk` updates, `chunk_chars`
/// characters at a time. Returns the stop reason.
fn stream_body(
    session_id: &Value,
    body: &str,
    chunk_chars: usize,
    delay: Duration,
    cancelled: &AtomicBool,
) -> &'static str {
    let chars: Vec<char> = body.chars().collect();
    let mut start = 0;
    while start < chars.len() {
        if cancelled.load(Ordering::SeqCst) {
            return "cancelled";
        }
        let end = (start + chunk_chars).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        start = end;

        send(json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": chunk },
                },
            },
        }));

        if !delay.is_zero() {
            thread::sleep(delay);
        }
    }

    "end_turn"
}

/// Build a deterministic markdown document of `blocks` self-contained, blank-line
/// separated top-level blocks (headings, wrapping prose, fenced code, lists,
/// quotes). `blocks` is the real markdown block count — i.e. the taffy-node
/// pressure that drives #57349 — not a raw line count.
fn generate_markdown(blocks: usize) -> String {
    let mut out = String::new();
    for i in 0..blocks {
        match i % 5 {
            0 => out.push_str(&format!("## Heading {i}\n\n")),
            1 => out.push_str(&format!(
                "Paragraph {i} with some **bold**, _italic_, `inline code`, and a \
                 [link](https://example.com) plus enough filler words to wrap across the \
                 agent panel a couple of times over.\n\n"
            )),
            2 => out.push_str(&format!("- bullet {i} a\n- bullet {i} b\n- bullet {i} c\n\n")),
            3 => out.push_str(&format!(
                "```rust\nfn block_{i}() -> usize {{ let x = {i}; x * 2 + 1 }}\n```\n\n"
            )),
            _ => out.push_str(&format!(
                "> quote {i} that wraps a little when the panel is narrow enough\n\n"
            )),
        }
    }
    out
}

/// Build a single giant plain-text paragraph (no markdown structure, no
/// newlines) of roughly the same character volume as `generate_markdown(lines)`.
/// This is the control for the "is it block count or message length?" test:
/// markdown parses this as ONE block (≈1 taffy node), so if scrolling this is
/// cheap while the structured version freezes, the cost is per-block taffy
/// layout, not total message size.
fn generate_plain(lines: usize) -> String {
    let mut out = String::new();
    for i in 0..lines {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&format!(
            "This is plain sentence number {i} with a handful of perfectly ordinary words and \
             no markdown structure whatsoever to speak of here."
        ));
    }
    out
}

fn extract_prompt_text(message: &Value) -> String {
    let Some(blocks) = message.pointer("/params/prompt").and_then(Value::as_array) else {
        return String::new();
    };
    let mut text = String::new();
    for block in blocks {
        if block.get("type").and_then(Value::as_str) == Some("text")
            && let Some(content) = block.get("text").and_then(Value::as_str)
        {
            text.push_str(content);
        }
    }
    text
}

fn respond(id: Option<Value>, result: Value) {
    send(json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(Value::Null),
        "result": result,
    }));
}

/// Write one newline-delimited JSON-RPC message to stdout. A write failure
/// means the client has gone away, so exit rather than spin.
fn send(message: Value) {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let write_result = serde_json::to_writer(&mut handle, &message)
        .map_err(std::io::Error::other)
        .and_then(|()| handle.write_all(b"\n"))
        .and_then(|()| handle.flush());
    if write_result.is_err() {
        std::process::exit(0);
    }
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
