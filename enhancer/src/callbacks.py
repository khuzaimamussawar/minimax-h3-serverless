from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import RuntimeConfig

EVENT_PATH = "/api/projects/v2/enhancer/pod/events"
H3_EVENT_PATH = "/api/projects/v2/h3/pod/events"
_PROCESS_STARTED = time.monotonic()
_JOB_ACTIVITY_SEEN = False


def event_url(control_url: str) -> str:
    """Use the same machine-to-machine callback ingress as H3 Pods."""
    url = control_url.rstrip("/")
    if url.endswith(H3_EVENT_PATH):
        return url
    if url.endswith(EVENT_PATH):
        return f"{url[:-len(EVENT_PATH)]}{H3_EVENT_PATH}"
    return f"{url}{H3_EVENT_PATH}"


def _seconds_timestamp(value: Any, fallback: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number <= 0:
        return fallback
    return number / 1000.0 if number >= 1e12 else number


def normalize_event_payload(config: RuntimeConfig, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize Enhancer callbacks to the H3 Pod lifecycle contract.

    H3 emits seconds-based timestamps and only emits worker_ready at boot. Older
    Enhancer server code emits an immediate worker_idle after worker_ready; drop
    that one startup duplicate so a new pod cannot trigger two dispatch races.
    """
    global _JOB_ACTIVITY_SEEN
    normalized = dict(payload)
    event_type = str(normalized.get("eventType") or normalized.get("event") or "").strip()
    normalized.setdefault("event", event_type)
    normalized.setdefault("eventType", event_type)
    normalized.setdefault("timestamp", time.time())

    if event_type.startswith("job_") and event_type not in {"job_cancelling"}:
        _JOB_ACTIVITY_SEEN = True

    if event_type == "worker_idle":
        # server.py historically emits worker_ready + worker_idle back-to-back
        # during boot. H3 emits only worker_ready; suppress only that pre-job
        # startup duplicate, never a real post-job idle event.
        if not _JOB_ACTIVITY_SEEN and time.monotonic() - _PROCESS_STARTED < 60.0:
            return None
        idle_since = _seconds_timestamp(normalized.get("idleSince"), time.time())
        timeout = max(0, int(normalized.get("idleTimeoutSeconds", config.idle_timeout_seconds) or 0))
        normalized["idleSince"] = idle_since
        normalized["idleTimeoutSeconds"] = timeout
        normalized.setdefault("terminateAfter", idle_since + timeout if timeout > 0 else None)

    if event_type in {"idle_expired", "worker_idle_timeout"}:
        idle_since = _seconds_timestamp(normalized.get("idleSince"), None)
        terminate_after = _seconds_timestamp(normalized.get("terminateAfter"), None)
        if idle_since is not None:
            normalized["idleSince"] = idle_since
        if terminate_after is None and idle_since is not None:
            terminate_after = idle_since + max(0, int(normalized.get("idleTimeoutSeconds", config.idle_timeout_seconds) or 0))
        if terminate_after is not None:
            normalized["terminateAfter"] = terminate_after

    return normalized


def post_event(config: RuntimeConfig, payload: dict[str, Any], timeout: float = 15.0) -> dict[str, Any] | None:
    normalized = normalize_event_payload(config, payload)
    if normalized is None:
        return None
    body = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    urls = [config.control_url]
    if config.fallback_control_url and config.fallback_control_url not in urls:
        urls.append(config.fallback_control_url)
    last_error: Exception | None = None
    for index, control_url in enumerate(urls):
        try:
            return _post_event_once(config, control_url, body, timeout)
        except RuntimeError as error:
            last_error = error
            if index >= len(urls) - 1 or not _should_try_fallback(error):
                raise
    if last_error:
        raise last_error
    return None


def _should_try_fallback(error: Exception) -> bool:
    text = str(error)
    return text.startswith("HTTP 401 ") or text.startswith("HTTP 403 ")


def _post_event_once(
    config: RuntimeConfig,
    control_url: str,
    body: bytes,
    timeout: float,
) -> dict[str, Any] | None:
    target_url = event_url(control_url)
    request = urllib.request.Request(
        target_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SceneBuilder-Enhancer-Pod/1.0",
            "Authorization": f"Bearer {config.pod_token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Match H3's best-effort callback contract: a successful HTTP
                # response is accepted even when an upstream layer returns a
                # non-JSON body. D1 state remains recoverable through the direct
                # pod polling loop, so callback response parsing must not make a
                # healthy GPU job look failed/noisy.
                return None
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"HTTP {error.code} {error.reason} from {target_url}: {response_body}") from error
