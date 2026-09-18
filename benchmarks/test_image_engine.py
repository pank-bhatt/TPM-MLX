import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from tpm_mlx.image_engine import MLXImageEngine, ImageGenerationMetrics, ImageResult


def test_parse_dimensions():
    # Aspect ratio presets
    assert MLXImageEngine.parse_dimensions("1:1") == (1024, 1024)
    assert MLXImageEngine.parse_dimensions("16:9") == (1024, 576)
    assert MLXImageEngine.parse_dimensions("9:16") == (576, 1024)
    assert MLXImageEngine.parse_dimensions("4:3") == (1024, 768)
    
    # Custom dimensions (rounded to multiples of 16)
    assert MLXImageEngine.parse_dimensions("512x512") == (512, 512)
    assert MLXImageEngine.parse_dimensions("800x600") == (800, 592)
    
    # Fallback for invalid format
    assert MLXImageEngine.parse_dimensions("invalid") == (1024, 1024)


def test_image_result_dataclass():
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    metrics = ImageGenerationMetrics(
        generation_time_s=4.5,
        steps=4,
        step_time_s=1.125,
        peak_memory_gb=6.2,
        width=64,
        height=64,
        seed=1234,
        model="flux2-klein-4b"
    )
    res = ImageResult(
        pil_image=img,
        b64_json="fake_base64",
        png_bytes=b"fake_bytes",
        metrics=metrics,
        revised_prompt="A revised prompt"
    )
    assert res.metrics.generation_time_s == 4.5
    assert res.metrics.steps == 4
    assert res.revised_prompt == "A revised prompt"


def test_server_image_endpoint():
    from fastapi.testclient import TestClient
    from tpm_mlx.server import app

    client = TestClient(app)
    
    # Mock MLXImageEngine.generate
    mock_metrics = ImageGenerationMetrics(
        generation_time_s=3.2,
        steps=4,
        step_time_s=0.8,
        peak_memory_gb=5.0,
        width=512,
        height=512,
        seed=42,
        model="flux2-klein-4b"
    )
    mock_res = ImageResult(
        pil_image=Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)),
        b64_json="mock_b64",
        png_bytes=b"mock_png",
        metrics=mock_metrics,
        revised_prompt="Synthesized visual prompt"
    )

    with patch("tpm_mlx.server._get_or_load_image_engine") as mock_get_engine:
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = mock_res
        mock_get_engine.return_value = mock_engine_instance

        resp = client.post("/v1/images/generations", json={
            "prompt": "A beautiful sunset over the mountains",
            "size": "512x512",
            "steps": 4,
            "response_format": "b64_json"
        })
        
        assert resp.status_code == 200
        data = resp.json()
        assert "created" in data
        assert len(data["data"]) == 1
        assert data["data"][0]["b64_json"] == "mock_b64"
        assert data["data"][0]["revised_prompt"] == "Synthesized visual prompt"
        assert data["tpm_metrics"]["generation_time_s"] == 3.2
        assert data["tpm_metrics"]["steps"] == 4


def test_resolve_input_image_path(tmp_path):
    from tpm_mlx.image_engine import _resolve_input_image_path

    # Test file that exists
    test_img = tmp_path / "test.png"
    test_img.write_bytes(b"fake_data")
    assert _resolve_input_image_path(str(test_img)) == test_img

    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        _resolve_input_image_path("/nonexistent/file/test_abc.png")


def test_server_image_upload_endpoint():
    from fastapi.testclient import TestClient
    from tpm_mlx.server import app

    client = TestClient(app)
    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

    resp = client.post(
        "/v1/upload_image",
        files={"file": ("test_upload.png", fake_png, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["url"].startswith("/images/uploads/")


def test_server_image_edit_endpoint():
    from fastapi.testclient import TestClient
    from tpm_mlx.server import app

    client = TestClient(app)

    mock_metrics = ImageGenerationMetrics(
        generation_time_s=5.1,
        steps=4,
        step_time_s=1.275,
        peak_memory_gb=6.5,
        width=512,
        height=768,
        seed=99,
        model="mlx-community/FLUX.2-klein-9B"
    )
    mock_res = ImageResult(
        pil_image=Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)),
        b64_json="mock_edit_b64",
        png_bytes=b"mock_png",
        metrics=mock_metrics,
        revised_prompt="Modified image prompt"
    )

    with patch("tpm_mlx.server._get_or_load_image_engine") as mock_get_engine:
        mock_engine_instance = MagicMock()
        mock_engine_instance.supports_editing = True
        mock_engine_instance.edit.return_value = mock_res
        mock_get_engine.return_value = mock_engine_instance

        resp = client.post("/v1/images/edits", json={
            "image": "/images/test.png",
            "prompt": "Add sunglasses and leather jacket",
            "guidance": 2.5,
            "steps": 4,
            "response_format": "b64_json"
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["b64_json"] == "mock_edit_b64"
        assert data["tpm_metrics"]["generation_time_s"] == 5.1
        assert data["tpm_metrics"]["steps"] == 4

