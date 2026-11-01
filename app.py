from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gradio as gr
import spaces


@spaces.GPU
def _zerogpu_runtime_compatibility_marker() -> str:
    """Required by the Space's free ZeroGPU runtime; never called by this app."""
    return "This simulator does not require GPU work."


PROVIDERS = ("OpenAI", "Anthropic")
DEFAULT_MODELS = {
    "OpenAI": "gpt-5.4-mini",
    "Anthropic": "claude-sonnet-4-6",
}
PROMPT_EXAMPLE = "Summarize the main point of this sentence in one line: Good error messages tell people what happened and what they can do next."

# These are example response shapes, not claims that every account receives
# the same status code or exact payload. Provider APIs and account settings vary.
SCENARIOS = {
    "OpenAI": (
        "Temporary rate limit with Retry-After",
        "Usage or spending limit",
        "Model temporarily overloaded",
        "Invalid API key",
        "Model name not found",
        "Invalid request",
        "Model refusal",
        "Answer stopped at output limit",
        "Tool handoff",
        "Connection lost",
    ),
    "Anthropic": (
        "Temporary rate limit with Retry-After",
        "Monthly spend cap reached",
        "Model overloaded",
        "Invalid API key",
        "Model name not found",
        "Invalid request",
        "Model refusal",
        "Answer stopped at token limit",
        "Tool handoff",
        "Connection lost",
    ),
}

CSS = """
:root{--ink:#20231f;--muted:#686e66;--paper:#f5f4ef;--line:#d8d9d1;--accent:#306c59;--warn:#7b4812;--danger:#8d2d2d}
.gradio-container{max-width:1120px!important;background:var(--paper)!important;color:var(--ink)!important}
#hero{padding:2.15rem 0 1.2rem;border-bottom:1px solid var(--line);margin-bottom:.6rem}
#hero h1{font-size:clamp(2.15rem,5vw,3.8rem);letter-spacing:-.045em;line-height:1.02;margin:.45rem 0 .85rem;color:var(--ink)}
.eyebrow{color:var(--accent);text-transform:uppercase;letter-spacing:.14em;font-size:.74rem;font-weight:750}
#hero p{max-width:760px;color:var(--muted);font-size:1.04rem;line-height:1.65;margin-bottom:.3rem}
#state-card{border:1px solid var(--line);background:#fff;border-radius:14px;padding:1.3rem 1.45rem;min-height:150px;box-shadow:0 8px 25px rgba(32,35,31,.035)}
.state-card--error{border-left:6px solid var(--danger)!important}.state-card--warning{border-left:6px solid #bd7a26!important}.state-card--success{border-left:6px solid var(--accent)!important}
.state-kicker{font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.12em;color:var(--accent)}
.state-title{font-size:1.35rem;line-height:1.25;margin:.48rem 0;color:var(--ink)}.state-copy{color:#3e443e;line-height:1.65;margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
.state-meta{margin-top:1rem;padding-top:.7rem;border-top:1px solid var(--line);color:var(--muted);font-size:.87rem;line-height:1.55}
.response-banner{display:flex;align-items:center;gap:.55rem;margin:-.15rem 0 .85rem;padding:.55rem .7rem;border-radius:7px;background:#f7f8f4;border:1px solid var(--line);color:#596159;font-size:.76rem;font-weight:650}.response-banner .signal{width:7px;height:7px;border-radius:50%;background:#4c9470;flex:0 0 auto}.response-banner.is-active{background:#fff8ea;border-color:#e4c98d;color:#7b4812}.response-banner.is-active .signal{background:#bd7a26;box-shadow:0 0 0 4px #bd7a2622;animation:pulse 1.1s ease-in-out infinite}.response-banner.is-error{background:#fff3f1;border-color:#e5b6b0;color:#8d2d2d}.response-steps{display:flex;gap:3px;margin-left:auto}.response-steps span{width:18px;height:3px;border-radius:3px;background:#d5d9d1}.response-steps span.on{background:#4c9470}.response-banner.is-active .response-steps span.on{background:#bd7a26}.response-banner.is-error .response-steps span.on{background:#b9574d}
.request-flow{margin:0 0 1rem;padding:.85rem .8rem .7rem;border:1px solid var(--line);border-radius:10px;background:#fafbf8;overflow:hidden}.flow-label{color:var(--muted);font-size:.67rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.65rem}.flow-track{display:flex;align-items:center;gap:.35rem}.flow-node{flex:1;min-width:0;padding:.55rem .35rem;border:1px solid #d8d9d1;border-radius:7px;text-align:center;background:#fff;color:#697068;font-size:.71rem;line-height:1.15}.flow-node b{display:block;color:#3d443d;font-size:.76rem;margin-bottom:.2rem}.flow-node i{display:inline-block;width:6px;height:6px;border-radius:50%;background:#c6cbc3}.flow-node.is-active{border-color:#bd7a26;background:#fff8ea;color:#7b4812}.flow-node.is-active i{background:#bd7a26;box-shadow:0 0 0 4px #bd7a2622}.flow-node.is-done{border-color:#4c9470;background:#f1faf4}.flow-node.is-done i{background:#4c9470}.flow-node.is-error{border-color:#b9574d;background:#fff3f1}.flow-node.is-error i{background:#b9574d}.flow-arrow{color:#a4aaa2;font-size:1rem;flex:0 0 auto}.flow-caption{margin-top:.55rem;color:var(--muted);font-size:.73rem}.flow-pulse{display:inline-block;width:5px;height:5px;border-radius:50%;background:#bd7a26;margin-left:.25rem;animation:pulse 1.1s ease-in-out infinite}@keyframes pulse{50%{opacity:.25;transform:scale(.65)}}
.state-details{margin-top:.9rem;color:var(--muted);font-size:.88rem}.state-details summary{cursor:pointer;color:var(--accent);font-weight:650}.state-details pre{white-space:pre-wrap;overflow-wrap:anywhere}
.model-output{margin-top:1rem;padding:1rem;border-radius:9px;background:#f4f5f1;border:1px solid var(--line);white-space:pre-wrap;overflow-wrap:anywhere;max-height:440px;overflow:auto;color:var(--ink);font: .95rem/1.65 system-ui,sans-serif}
.note,.privacy{color:var(--muted);font-size:.91rem;line-height:1.55}.privacy{padding:.8rem 1rem;border-left:4px solid #bd7a26;background:#fffaf1;border-radius:4px}
.hint-card{border-top:1px solid var(--line);margin-top:1.1rem;padding-top:1rem;color:#4b514b;font-size:.93rem;line-height:1.6}.hint-card strong{color:var(--ink)}
button.primary{background:var(--accent)!important}footer{border-top:1px solid var(--line);margin-top:1rem;padding-top:1rem;color:var(--muted);font-size:.88rem}
@media(max-width:640px){#hero{padding-top:1.3rem}#state-card{padding:1.05rem;min-height:0}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;transition-duration:.01ms!important}}
"""


@dataclass
class Outcome:
    title: str
    message: str
    kind: str = "success"
    role: str = "status"
    next_step: str = ""
    technical: str = ""
    output: str = ""


def _esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def render_outcome(outcome: Outcome) -> str:
    active = any(word in outcome.title for word in ("Sending", "Streaming", "Receiving", "Retry countdown", "Waiting", "Finalizing"))
    if active:
        banner_class, banner_text, active_steps = "is-active", "Response is coming in", 3
    elif outcome.kind == "error":
        banner_class, banner_text, active_steps = "is-error", "Request stopped · next step available", 2
    elif outcome.kind == "warning":
        banner_class, banner_text, active_steps = "", "Response received · action needed", 4
    else:
        banner_class, banner_text, active_steps = "", "Response complete", 4
    step_markup = "".join(f'<span class="{"on" if i < active_steps else ""}"></span>' for i in range(4))
    banner = f'<div class="response-banner {banner_class}"><span class="signal"></span><span>{_esc(banner_text)}</span><div class="response-steps">{step_markup}</div></div>'
    if outcome.kind == "error":
        provider_state, stream_state, caption = "is-error", "is-error", "The provider or network returned a failure."
    elif outcome.kind == "warning":
        provider_state, stream_state, caption = "is-active", "is-active", "The request is moving through the provider response path."
    else:
        provider_state, stream_state, caption = "is-done", "is-done", "The request completed successfully."
    if "Sending" in outcome.title or "Waiting" in outcome.title:
        client_state, gateway_state = "is-active", "is-active"
    elif outcome.kind == "success":
        client_state, gateway_state = "is-done", "is-done"
    else:
        client_state, gateway_state = "is-done", "is-done"
    pulse = '<span class="flow-pulse"></span>' if outcome.kind == "warning" else ""
    flow = f'''<div class="request-flow"><div class="flow-label">Request flow</div><div class="flow-track">
      <div class="flow-node {client_state}"><i></i><b>Client</b>input</div><span class="flow-arrow">→</span>
      <div class="flow-node {gateway_state}"><i></i><b>API gateway</b>route</div><span class="flow-arrow">→</span>
      <div class="flow-node {provider_state}"><i></i><b>Provider</b>model</div><span class="flow-arrow">→</span>
      <div class="flow-node {stream_state}"><i></i><b>Response</b>stream</div></div>
      <div class="flow-caption">{_esc(caption)}{pulse}</div></div>'''
    technical = ""
    if outcome.technical:
        technical = (
            '<details class="state-details"><summary>Provider details</summary>'
            f"<pre>{_esc(outcome.technical)}</pre></details>"
        )
    answer = ""
    if outcome.output:
        answer = f'<pre class="model-output">{_esc(outcome.output)}</pre>'
    meta = f'<div class="state-meta"><strong>Next step</strong><br>{_esc(outcome.next_step)}</div>' if outcome.next_step else ""
    return f"""<section id="state-card" class="state-card--{_esc(outcome.kind)}">
      {banner}{flow}<div role="{_esc(outcome.role)}" aria-atomic="true">
        <div class="state-kicker">{_esc(outcome.kind)}</div>
        <h2 class="state-title">{_esc(outcome.title)}</h2>
        <p class="state-copy">{_esc(outcome.message)}</p>
      </div>{meta}{answer}{technical}
    </section>"""


def _body_fields(provider: str, body: dict[str, Any]) -> tuple[str, str, str]:
    err = body.get("error")
    if isinstance(err, dict):
        typ = str(err.get("type") or body.get("type") or "")
        code = str(err.get("code") or err.get("details", {}).get("error_code", "") or "")
        # Never show a raw provider message here: it can echo user input.
        return typ, code, str(body.get("request_id") or body.get("requestId") or "")
    if provider == "OpenAI":
        err = body.get("error") if isinstance(body.get("error"), dict) else {}
        return str(err.get("type", "")), str(err.get("code", "")), str(body.get("id", ""))
    return "", "", str(body.get("request_id", ""))


def _duration_seconds(value: str) -> int | None:
    if not value:
        return None
    try:
        return max(0, int(float(value.strip())))
    except (ValueError, TypeError):
        try:
            when = parsedate_to_datetime(value)
            return max(0, int(when.timestamp() - time.time()))
        except (ValueError, TypeError, OverflowError):
            return None


def classify_http_error(provider: str, status: int, headers: Any, body: dict[str, Any]) -> Outcome:
    err = body.get("error", {}) if isinstance(body.get("error"), dict) else {}
    err_details = err.get("details", {}) if isinstance(err.get("details"), dict) else {}
    typ, code, req_id = _body_fields(provider, body)
    message_for_detection = " ".join(str(x) for x in [err.get("message", ""), code, err_details.get("error_code", "")]).lower()
    retry_header = headers.get("retry-after") or headers.get("Retry-After") or ""
    retry_seconds = _duration_seconds(str(retry_header))
    facts = [f"HTTP {status}"]
    if typ:
        facts.append(f"type: {typ}")
    if code:
        facts.append(f"code: {code}")
    if retry_header:
        facts.append(f"Retry-After: {retry_header}")
    if req_id:
        facts.append(f"request ID: {req_id}")
    technical = "\n".join(facts)

    spend_codes = {
        "credit_balance_exhausted", "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
        "enforced_spend_limit_reached",
    }
    is_spend = code in spend_codes or "spend limit" in message_for_detection or "spending limit" in message_for_detection or "usage limit" in message_for_detection or "monthly api usage threshold" in message_for_detection
    if status == 429 and is_spend:
        return Outcome("Your account limit was reached", "We couldn't send this request because the account has reached its usage or spending limit. Waiting and trying the same request again will not fix it.", "error", "alert", "Check your plan, credits, usage limit or billing settings. Your prompt is still here, and no request will be retried automatically.", technical)
    if provider == "Anthropic" and status == 400 and is_spend:
        return Outcome("Your account limit was reached", "We couldn't send this request because the account has reached its spending limit. This is an account setting, not a problem with your prompt.", "error", "alert", "Review the account's billing and spending settings before trying again.", technical)
    if status == 429:
        if retry_seconds is not None:
            return Outcome("The service is busy right now", f"We couldn't complete that request because the service asked us to wait at least {retry_seconds} seconds. Nothing was lost.", "warning", "status", "Wait the requested time, then try once more. If this keeps happening, reduce request volume or try again later.", technical)
        return Outcome("Too many requests right now", "The service is receiving more requests than it can handle at the moment. We don't have a reliable wait time, so we won't invent a countdown.", "warning", "status", "Wait a moment and try again once. If it continues, check your account usage or reduce request volume.", technical)
    if status in (500, 502, 503, 504, 529):
        retry_note = f" The service asked us to wait at least {retry_seconds} seconds." if retry_seconds is not None else ""
        return Outcome("The service couldn't finish that request", "Something went wrong on the model service while processing your request." + retry_note, "error", "alert", "Keep your prompt and try again after a short delay. We won't retry automatically because the request may already have reached the service.", technical)
    if status in (401, 403):
        return Outcome("We couldn't verify your access", "The access key was not accepted, or it does not have permission to use this model.", "error", "alert", "Check the key, project, workspace and model access. For safety, enter a new key rather than retrying the same rejected one.", technical)
    if status == 402:
        return Outcome("Billing needs attention", "We couldn't send this request because billing is not ready for this account.", "error", "alert", "Review the account's payment method, credits or billing status, then try again.", technical)
    if status == 404 or "model_not_found" in code or "not_found_error" in typ:
        return Outcome("That model isn't available", "We couldn't find this model for the selected account, or your key doesn't have access to it.", "error", "alert", "Check the model ID and choose a model enabled for this account.", technical)
    if status in (400, 413, 422):
        title = "That request needs a small change"
        message = "We couldn't send this request as written. A setting, model option or input size may not be supported."
        next_step = "Check the details below, then adjust the model or request before trying again."
        if status == 413:
            title, message, next_step = "That request is too large", "We couldn't send this request because its input is bigger than the service allows.", "Shorten the prompt or reduce attached content, then submit again."
        return Outcome(title, message, "error", "alert", next_step, technical)
    return Outcome("We couldn't complete that request", f"The request did not finish normally. We've kept the technical details below so the issue can be investigated without exposing the full response.", "error", "alert", "Keep your prompt and try again once. If it continues, share the request ID with the app administrator.", technical)


def classify_success(provider: str, body: dict[str, Any], headers: Any, elapsed: float) -> Outcome:
    request_id = str(headers.get("x-request-id") or headers.get("request-id") or headers.get("anthropic-request-id") or body.get("id") or body.get("request_id") or "")
    usage = body.get("usage", {}) if isinstance(body.get("usage"), dict) else {}
    usage_line = []
    if usage.get("input_tokens") is not None:
        usage_line.append(f"input tokens: {usage['input_tokens']}")
    if usage.get("output_tokens") is not None:
        usage_line.append(f"output tokens: {usage['output_tokens']}")
    facts = [f"HTTP 200 · {elapsed:.1f} seconds"]
    if body.get("model"):
        facts.append(f"model: {body['model']}")
    if usage_line:
        facts.extend(usage_line)
    if request_id:
        facts.append(f"request ID: {request_id}")

    texts: list[str] = []
    refusal = False
    tool_use = False
    if provider == "OpenAI":
        status = body.get("status")
        incomplete = body.get("incomplete_details") or {}
        if status == "failed" or body.get("error"):
            error = body.get("error") or {}
            return Outcome("We couldn't complete that response", "The request reached the model service, but it could not finish generating an answer.", "error", "alert", "Keep your input and try again once. If it repeats, share the request ID with the app administrator.", "\n".join(facts + [f"response status: {status}", f"error code: {error.get('code', '')}"]))
        for item in body.get("output", []) or []:
            if item.get("type") == "function_call":
                tool_use = True
            for block in item.get("content", []) or []:
                if block.get("type") == "refusal":
                    refusal = True
                    texts.append(str(block.get("refusal", "The model declined this request.")))
                elif block.get("type") == "output_text":
                    texts.append(str(block.get("text", "")))
        if incomplete:
            facts.append(f"incomplete reason: {incomplete.get('reason', 'unspecified')}")
    else:
        stop_reason = body.get("stop_reason") or ""
        stop_details = body.get("stop_details") or {}
        refusal = stop_reason == "refusal" or (isinstance(stop_details, dict) and stop_details.get("type") == "refusal")
        for block in body.get("content", []) or []:
            if block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                tool_use = True
        if stop_reason:
            facts.append(f"stop reason: {stop_reason}")

    output_text = "\n\n".join(piece for piece in texts if piece).strip()
    technical = "\n".join(facts)
    if refusal:
        return Outcome("The model chose not to answer", "The request reached the model, but it declined to provide the requested content. This is a response, not a connection failure.", "warning", "status", "Try a permitted adjacent request or rephrase the goal. Repeating the same request may produce the same result.", technical, output_text or "The model marked this response as a refusal.")
    if tool_use:
        return Outcome("The model needs another step", "The model asked the app to use a tool, but this demo stopped before running anything.", "warning", "status", "In a production app, validate the tool and arguments, ask for authorization when needed, then continue the request.", technical, output_text or "A tool handoff was received. No tool was executed.")
    incomplete_reason = (body.get("incomplete_details") or {}).get("reason") if provider == "OpenAI" else body.get("stop_reason")
    if incomplete_reason in ("max_output_tokens", "max_tokens", "model_context_window_exceeded"):
        return Outcome("The answer stopped early", "The model reached an output limit before finishing. The text below is partial, not a complete answer.", "warning", "status", "Keep the partial answer, ask the app to continue, shorten the request or raise the output limit when appropriate.", technical, output_text or "No text was returned before the limit.")
    if not output_text:
        return Outcome("The request finished without text", "The API returned a successful response, but this example could not find user facing text in it.", "warning", "status", "Inspect the response type. It may contain a non text item that your interface needs to handle.", technical, "No text output found.")
    return Outcome("Your answer is ready", "The request completed successfully. You can continue from this answer or send a follow-up.", "success", "status", "Review the answer before relying on it, especially for important decisions.", technical, output_text)


def _request_live(provider: str, model: str, api_key: str, prompt: str, max_tokens: int) -> Outcome:
    if provider not in PROVIDERS:
        return Outcome("Choose a supported provider", "Select OpenAI or Anthropic to make a live request.", "error", "alert", "Choose a provider from the list.")
    model = (model or "").strip()
    api_key = (api_key or "").strip()
    prompt = (prompt or "").strip()
    if not model:
        return Outcome("Enter a model ID", "The model field is empty.", "error", "alert", "Enter a model name enabled for your provider account.")
    if not api_key:
        return Outcome("Enter an API key", "A live request needs a provider key. The example simulator does not.", "error", "alert", "Add a key for one request, or use the simulator tab instead.")
    if not prompt:
        return Outcome("Enter a prompt", "The live request needs text to send.", "error", "alert", "Write a short, non confidential test prompt.")

    if provider == "OpenAI":
        url = "https://api.openai.com/v1/responses"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "input": prompt, "max_output_tokens": int(max_tokens)}
    else:
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        payload = {"model": model, "max_tokens": int(max_tokens), "messages": [{"role": "user", "content": prompt}]}
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urlopen(request, timeout=45) as response:
            response_headers = response.headers
            raw = response.read(2_000_000)
            status = response.status
        body = json.loads(raw.decode("utf-8"))
        if not isinstance(body, dict):
            return Outcome("The provider returned an unexpected response", "The request returned successfully, but the body was not a JSON object the demo recognizes.", "warning", "status", "Check the provider's current response format.", f"HTTP {status}")
        return classify_success(provider, body, response_headers, time.monotonic() - started)
    except HTTPError as exc:
        raw = exc.read(256_000)
        try:
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                body = {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        return classify_http_error(provider, exc.code, exc.headers, body)
    except (TimeoutError, URLError, OSError) as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
            return Outcome("The request took too long", "We didn't receive a final answer before the 45-second limit. The service may or may not have finished the work.", "error", "alert", "Keep your prompt. Check request history before trying again, especially for work that could have completed or incurred a charge.", "No HTTP response · timeout")
        return Outcome("We lost the connection", "The request left the app, but no response came back. We can't tell from this screen whether the service received it.", "error", "alert", "Check request history before replaying the request, then try again when the connection is stable.", "No HTTP response · network connection error")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return Outcome("We couldn't read the response", "The service answered, but the response was not in a format the app understands.", "error", "alert", "No blank answer was shown. Try again once, and report this issue if it repeats.", "Response parsing error")
    except Exception:
        # Avoid returning exception strings; third party exceptions can contain request data.
        return Outcome("Something went wrong in the app", "The app hit an unexpected problem and hid the technical exception so it would not expose request details.", "error", "alert", "Try a short prompt once more. If it repeats, report the model name and request ID without sharing your key.", "Unexpected client exception hidden")


def live_request(provider: str, model: str, api_key: str, prompt: str, max_tokens: int):
    # This is intentionally a generator so Gradio can show progress while the
    # bounded network request is in flight. It never retries automatically.
    yield render_outcome(Outcome("Sending your request", "Your request is on its way. The app will not retry automatically.", "warning", "status", "Wait for the result. The key field is cleared as soon as this request finishes.")), ""
    result = _request_live(provider, model, api_key, prompt, max_tokens)
    yield render_outcome(result), ""


SIM_FIXTURES: dict[str, dict[str, dict[str, Any]]] = {
    "OpenAI": {
        "Temporary rate limit with Retry-After": {"status": 429, "headers": {"Retry-After": "12"}, "body": {"error": {"type": "rate_limit_error", "code": "slow_down", "message": "Request rate increased too quickly."}}},
        "Usage or spending limit": {"status": 429, "headers": {}, "body": {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted", "message": "Credits exhausted."}}},
        "Model temporarily overloaded": {"status": 503, "headers": {"Retry-After": "8"}, "body": {"error": {"type": "service_unavailable_error", "code": "server_is_overloaded", "message": "Model temporarily overloaded."}}},
        "Invalid API key": {"status": 401, "headers": {}, "body": {"error": {"type": "invalid_request_error", "code": "invalid_api_key", "message": "Invalid key."}}},
        "Model name not found": {"status": 404, "headers": {}, "body": {"error": {"type": "invalid_request_error", "code": "model_not_found", "message": "No model with this ID."}}},
        "Invalid request": {"status": 400, "headers": {}, "body": {"error": {"type": "invalid_request_error", "code": "unsupported_value", "message": "The parameter is not supported."}}},
        "Model refusal": {"status": 200, "headers": {}, "body": {"id": "resp_demo", "status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "I cannot help with that request."}]}]}},
        "Answer stopped at output limit": {"status": 200, "headers": {}, "body": {"id": "resp_demo", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [{"type": "message", "content": [{"type": "output_text", "text": "The first part of the answer…"}]}]}},
        "Tool handoff": {"status": 200, "headers": {}, "body": {"id": "resp_demo", "status": "completed", "output": [{"type": "function_call", "name": "lookup_order", "arguments": '{"id":"example"}'}]}},
        "Connection lost": {"network": True},
    },
    "Anthropic": {
        "Temporary rate limit with Retry-After": {"status": 429, "headers": {"retry-after": "12"}, "body": {"type": "error", "error": {"type": "rate_limit_error", "message": "Request rate limit reached."}, "request_id": "req_demo"}},
        "Monthly spend cap reached": {"status": 429, "headers": {}, "body": {"type": "error", "error": {"type": "rate_limit_error", "message": "Monthly API usage cap reached.", "details": {"error_code": "enforced_spend_limit_reached"}}, "request_id": "req_demo"}},
        "Model overloaded": {"status": 529, "headers": {}, "body": {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded."}, "request_id": "req_demo"}},
        "Invalid API key": {"status": 401, "headers": {}, "body": {"type": "error", "error": {"type": "authentication_error", "message": "Invalid key."}, "request_id": "req_demo"}},
        "Model name not found": {"status": 404, "headers": {}, "body": {"type": "error", "error": {"type": "not_found_error", "message": "Model not found."}, "request_id": "req_demo"}},
        "Invalid request": {"status": 400, "headers": {}, "body": {"type": "error", "error": {"type": "invalid_request_error", "message": "Malformed request."}, "request_id": "req_demo"}},
        "Model refusal": {"status": 200, "headers": {}, "body": {"id": "msg_demo", "stop_reason": "refusal", "content": [{"type": "text", "text": "I cannot help with that request."}]}},
        "Answer stopped at token limit": {"status": 200, "headers": {}, "body": {"id": "msg_demo", "stop_reason": "max_tokens", "content": [{"type": "text", "text": "The first part of the answer…"}]}},
        "Tool handoff": {"status": 200, "headers": {}, "body": {"id": "msg_demo", "stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "toolu_demo", "name": "lookup_order", "input": {"id": "example"}}]}},
        "Connection lost": {"network": True},
    },
}


def _sim_step(title: str, message: str, *, output: str = "", technical: str = "") -> str:
    """Render one visible step of the simulated request lifecycle."""
    return render_outcome(Outcome(
        title, message, "warning", "status",
        "This is a simulated response. No key is used and no provider is contacted.",
        technical, output,
    ))


def _stream_simulated_output(result: Outcome, technical: str):
    """Yield a few partial response states so the simulator feels like a stream."""
    words = (result.output or "").split()
    if not words:
        return
    partial = []
    for word in words:
        partial.append(word)
        yield render_outcome(Outcome(
            "Receiving the provider response",
            "Text is arriving in chunks. The interface keeps the partial answer visible.",
            "warning", "status",
            "The final state will appear when the simulated stream ends.",
            technical, " ".join(partial),
        ))
        time.sleep(0.045)


def simulate(provider: str, scenario: str):
    """Run a visible request lifecycle instead of returning one static card."""
    fixture = SIM_FIXTURES.get(provider, {}).get(scenario)
    if not fixture:
        yield render_outcome(Outcome("Choose a scenario", "Pick a provider and one of its example outcomes.", "warning", "status", "Nothing is sent to a model provider."))
        return

    yield _sim_step(
        "Sending simulated request",
        f"Preparing the {provider} example for “{scenario}”.",
        technical="SIMULATION START · no key used and no API call made",
    )
    time.sleep(0.35)

    if fixture.get("network"):
        yield _sim_step("Waiting for provider response", "The simulated connection is open, but no response is arriving.", technical="No HTTP response yet")
        time.sleep(0.65)
        result = Outcome("We could not reach the provider", "No HTTP response arrived. A network interruption does not prove the remote request did not finish.", "error", "alert", "Keep the prompt. Check the provider's request history before replaying work that could have completed.", "SIMULATED RESPONSE · no key used and no API call made\nNo HTTP response · connection interruption")
        yield render_outcome(result)
        return

    status = fixture["status"]
    if status == 429 and fixture["headers"].get("Retry-After"):
        retry_after = fixture["headers"]["Retry-After"]
        technical = f"HTTP 429\nRetry-After: {retry_after}\nSIMULATED RESPONSE · no key used and no API call made"
        yield _sim_step("Connecting to the service", f"Sending the request to the simulated {provider} service…", technical="Connection opened\nSIMULATED RESPONSE · no key used and no API call made")
        time.sleep(0.45)
        yield _sim_step("Waiting for a response", "The service is processing the request. The response has not arrived yet…", technical="Request in progress\nNo error shown yet")
        time.sleep(0.7)
        yield _sim_step("Response received", "The service replied and asked the app to slow down before trying again.", technical=technical)
        time.sleep(0.35)
        # Keep the demo quick while making the countdown and retry decision visible.
        for left in (3, 2, 1):
            yield _sim_step("Retry countdown", f"Retrying in {left}s (accelerated demo countdown; provider hint was {retry_after}s).", technical=technical)
            time.sleep(0.3)
    elif status >= 400:
        yield _sim_step("Connecting to the service", f"Sending the request to the simulated {provider} service…", technical="Connection opened\nSIMULATED RESPONSE · no key used and no API call made")
        time.sleep(0.45)
        yield _sim_step("Waiting for a response", "The service is processing the request. The response has not arrived yet…", technical="Request in progress\nNo error shown yet")
        time.sleep(0.7)
        yield _sim_step("Response received", f"The simulated service replied. The app is checking what happened before showing a message.", technical=f"HTTP {status}\nSIMULATED RESPONSE · no key used and no API call made")
        time.sleep(0.45)
    else:
        yield _sim_step("Streaming response", "The simulated provider accepted the request. The answer will arrive in chunks.", technical="HTTP 200 · stream opened\nSIMULATED RESPONSE · no key used and no API call made")
        time.sleep(0.3)

    if fixture["status"] >= 400:
        result = classify_http_error(provider, fixture["status"], fixture["headers"], fixture["body"])
    else:
        result = classify_success(provider, fixture["body"], fixture["headers"], 0.8)
    result.technical = "SIMULATED RESPONSE · no key used and no API call made\n" + (result.technical or f"HTTP {fixture['status']}")
    if result.output:
        yield from _stream_simulated_output(result, result.technical)
    else:
        yield _sim_step("Finalizing response", "The provider state has been received. Rendering the user-facing explanation.", technical=result.technical)
        time.sleep(0.25)
    yield render_outcome(result)


with gr.Blocks(title="AI API Error States", analytics_enabled=False) as demo:
    gr.HTML("""<header id="hero"><div class="eyebrow">A small interface lab</div>
      <h1>When the model does not answer</h1>
      <p>Try a real text request with your own provider key, or inspect realistic example responses without sending anything.</p></header>""")
    with gr.Tabs():
        with gr.Tab("Try a live request"):
            gr.Markdown("One short text request, with no automatic retries. You control the model and output limit.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    live_provider = gr.Dropdown(PROVIDERS, value="OpenAI", label="Provider")
                    live_model = gr.Textbox(value=DEFAULT_MODELS["OpenAI"], label="Model ID", info="Enter a model enabled for your provider account.")
                    api_key = gr.Textbox(label="Provider API key", type="password", placeholder="Paste a temporary key for this request")
                    prompt = gr.Textbox(value=PROMPT_EXAMPLE, label="Prompt", lines=4, max_lines=8)
                    max_tokens = gr.Slider(32, 2048, value=400, step=32, label="Maximum output tokens", info="This provider setting can affect your API charges.")
                    live_button = gr.Button("Send one live request", variant="primary")
                    gr.HTML("""<div class="privacy"><strong>Before you send:</strong> this public Space sends your key and prompt through its hosted server to the provider you choose. The app code does not save or log the key and clears the key field as soon as the request starts, but hosted infrastructure is outside the app's control. Use a temporary key with a low spending limit. Do not send confidential prompts. The provider may charge your account. For stronger privacy, run the code on your own computer.</div>""")
                with gr.Column(scale=2):
                    live_result = gr.HTML(render_outcome(Outcome("Ready for one request", "Choose a provider, check the model ID, and enter a prompt. Nothing is sent until you press the button.", "success", "status", "Use a temporary, restricted key. The app will clear the field after the attempt.")))
            live_provider.change(lambda p: gr.update(value=DEFAULT_MODELS.get(p, "")), inputs=live_provider, outputs=live_model)
            live_button.click(live_request, inputs=[live_provider, live_model, api_key, prompt, max_tokens], outputs=[live_result, api_key], show_progress="minimal")
        with gr.Tab("Explore example responses"):
            gr.Markdown("Choose an example provider response. These are fixtures based on documented response shapes. No key or network call is used.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    sim_provider = gr.Dropdown(PROVIDERS, value="OpenAI", label="Provider")
                    sim_case = gr.Dropdown(SCENARIOS["OpenAI"], value=SCENARIOS["OpenAI"][0], label="Example response")
                    sim_button = gr.Button("Show this state", variant="primary")
                    gr.Markdown("Examples cover status codes, retry hints, spending limits, refusals, incomplete answers, tool handoffs and connection loss.", elem_classes=["note"])
                with gr.Column(scale=2):
                    sim_result = gr.HTML(render_outcome(Outcome("Choose an example", "The panel will show the user facing message, the next step and the provider details.", "success", "status", "Start with a temporary rate limit or compare a refusal with a server error.")))
            sim_provider.change(lambda p: gr.update(choices=SCENARIOS.get(p, ()), value=SCENARIOS.get(p, ("",))[0]), inputs=sim_provider, outputs=sim_case)
            sim_button.click(simulate, inputs=[sim_provider, sim_case], outputs=sim_result, show_progress="hidden")
    gr.HTML("""<div class="hint-card"><strong>The useful distinction</strong><br>
      A temporary limit can justify a wait. A known spending cap needs an account change. A refusal, partial answer or tool handoff is a response state, not an HTTP crash.</div>
      <footer>Updates are marked with <code>role="status"</code> or <code>role="alert"</code>. The example tab never contacts a provider.</footer>""")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        css=CSS,
        server_name="0.0.0.0",
        server_port=7860,
    )
