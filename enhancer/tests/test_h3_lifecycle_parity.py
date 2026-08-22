from pathlib import Path


SERVER = (Path(__file__).parents[1] / "src" / "server.py").read_text(encoding="utf-8")


def test_server_callbacks_include_h3_seconds_timestamp():
    event = SERVER[SERVER.index("def _event"):SERVER.index("def _post_event_background")]
    assert '"timestamp": time.time()' in event


def test_boot_emits_worker_ready_once_without_startup_worker_idle():
    boot = SERVER[SERVER.index("def _boot"):SERVER.index("def _idle_monitor")]
    assert '_event("worker_ready"' in boot
    assert '_event("worker_idle"' not in boot
    assert "readyAt=time.time()" in boot
    assert "_IDLE_SINCE = None" in boot
    assert "_IDLE_SINCE = time.time()" not in boot


def test_encoder_qualification_is_job_specific():
    boot = SERVER[SERVER.index("def _boot"):SERVER.index("def _idle_monitor")]
    run_job = SERVER[SERVER.index("def _run_job"):SERVER.index("def _require_auth")]
    assert "qualify_gpu(require_nvenc=False)" in boot
    assert 'normalize_video_encoder(settings) == "nvenc"' in run_job
    assert "qualify_gpu(require_nvenc=True)" in run_job


def test_post_job_idle_has_explicit_terminate_after():
    run_job = SERVER[SERVER.index("def _run_job"):SERVER.index("def _require_auth")]
    assert '_event("worker_idle"' in run_job
    assert "_IDLE_SINCE = time.time()" in run_job
    assert "terminateAfter=(idle_since + timeout)" in run_job


def test_idle_expired_uses_seconds_and_explicit_deadline():
    monitor = SERVER[SERVER.index("def _idle_monitor"):SERVER.index("def _capabilities")]
    assert '_event("idle_expired"' in monitor
    assert "idleSince=idle_since" in monitor
    assert "terminateAfter=idle_since + timeout" in monitor
    assert "int(idle_since * 1000)" not in monitor
