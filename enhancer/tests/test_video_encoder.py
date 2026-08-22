from unittest.mock import patch

from enhancer.src import video_encoder


def test_nvenc_is_default_and_keeps_cq_control():
    with patch.object(video_encoder, "_ensure_encoder_available", return_value=None):
        args = video_encoder.video_encoder_args({"nvencCq": 19})
    assert video_encoder.normalize_video_encoder({}) == "nvenc"
    assert args[args.index("-c:v") + 1] == "hevc_nvenc"
    assert args[args.index("-cq") + 1] == "19"
    assert "libx265" not in args


def test_x265_is_exclusive_and_keeps_crf_control():
    with patch.object(video_encoder, "_ensure_encoder_available", return_value=None):
        args = video_encoder.video_encoder_args({"videoEncoder": "x265", "x265Crf": 14, "nvencCq": 20})
    assert video_encoder.normalize_video_encoder({"videoEncoder": "x265"}) == "x265"
    assert args[args.index("-c:v") + 1] == "libx265"
    assert args[args.index("-crf") + 1] == "14"
    assert "hevc_nvenc" not in args
    assert "-cq" not in args


def test_encoder_quality_is_clamped_to_admin_supported_12_25_range():
    with patch.object(video_encoder, "_ensure_encoder_available", return_value=None):
        low_nvenc = video_encoder.video_encoder_args({"videoEncoder": "nvenc", "nvencCq": 1})
        high_nvenc = video_encoder.video_encoder_args({"videoEncoder": "nvenc", "nvencCq": 40})
        low_x265 = video_encoder.video_encoder_args({"videoEncoder": "x265", "x265Crf": 1})
        high_x265 = video_encoder.video_encoder_args({"videoEncoder": "x265", "x265Crf": 40})
    assert low_nvenc[low_nvenc.index("-cq") + 1] == "12"
    assert high_nvenc[high_nvenc.index("-cq") + 1] == "25"
    assert low_x265[low_x265.index("-crf") + 1] == "12"
    assert high_x265[high_x265.index("-crf") + 1] == "25"


def test_encoder_aliases_and_failure_codes_are_stable():
    assert video_encoder.normalize_video_encoder({"videoEncoder": "hevc_nvenc"}) == "nvenc"
    assert video_encoder.normalize_video_encoder({"videoEncoder": "libx265"}) == "x265"
    assert video_encoder.video_encoder_failure_code({"videoEncoder": "nvenc"}) == "NVENC_ENCODE_FAILED"
    assert video_encoder.video_encoder_failure_code({"videoEncoder": "x265"}) == "X265_ENCODE_FAILED"
