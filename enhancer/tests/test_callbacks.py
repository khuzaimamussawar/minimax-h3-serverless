import importlib
from unittest.mock import patch

from enhancer.src import callbacks
from enhancer.src.callbacks import EVENT_PATH, H3_EVENT_PATH, event_url
from enhancer.src.config import RuntimeConfig


def _config() -> RuntimeConfig:
    return RuntimeConfig(
        worker_id="enh-test",
        pod_token="token",
        control_url="https://scene-builder.example.com/api/projects/v2/h3/pod/events",
        fallback_control_url="",
        r2_bucket="bucket",
        r2_endpoint="https://r2.example.com",
        r2_access_key="access",
        r2_secret_key="secret",
        r2_region="auto",
        r2_public_url="",
        port=8000,
        idle_timeout_seconds=60,
        service_kind="enhancer_fast",
        debug=False,
    )


def test_enhancer_callbacks_use_h3_machine_ingress_from_origin():
    assert event_url("https://scene-builder.example.com") == (
        "https://scene-builder.example.com" + H3_EVENT_PATH
    )


def test_enhancer_callback_path_is_rewritten_to_h3_ingress():
    assert event_url("https://scene-builder.example.com" + EVENT_PATH) == (
        "https://scene-builder.example.com" + H3_EVENT_PATH
    )


def test_h3_callback_path_is_left_unchanged():
    target = "https://scene-builder.example.com" + H3_EVENT_PATH
    assert event_url(target) == target


def test_callback_payload_adds_h3_seconds_timestamp():
    module = importlib.reload(callbacks)
    payload = module.normalize_event_payload(_config(), {
        "event": "worker_ready",
        "eventType": "worker_ready",
        "workerId": "enh-test",
    })
    assert payload is not None
    assert 1_000_000_000 < payload["timestamp"] < 100_000_000_000


def test_boot_worker_idle_is_suppressed_until_real_job_activity():
    module = importlib.reload(callbacks)
    startup_idle = module.normalize_event_payload(_config(), {
        "event": "worker_idle",
        "eventType": "worker_idle",
        "workerId": "enh-test",
        "idleSince": 1_700_000_000_000,
        "idleTimeoutSeconds": 60,
    })
    assert startup_idle is None

    module.normalize_event_payload(_config(), {
        "event": "job_completed",
        "eventType": "job_completed",
        "workerId": "enh-test",
        "jobId": "job-1",
    })
    real_idle = module.normalize_event_payload(_config(), {
        "event": "worker_idle",
        "eventType": "worker_idle",
        "workerId": "enh-test",
        "idleSince": 1_700_000_000_000,
        "idleTimeoutSeconds": 60,
    })
    assert real_idle is not None
    assert real_idle["idleSince"] == 1_700_000_000
    assert real_idle["terminateAfter"] == 1_700_000_060


def test_successful_non_json_callback_response_is_best_effort_like_h3():
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"<html><title>Sign in</title></html>"

    module = importlib.reload(callbacks)
    with patch.object(module.urllib.request, "urlopen", return_value=FakeResponse()):
        assert module._post_event_once(
            _config(),
            "https://scene-builder.example.com/api/projects/v2/h3/pod/events",
            b"{}",
            1.0,
        ) is None
