from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _scripts_dir() -> Path:
    p = Path("/scripts")
    if p.is_dir():
        return p
    return Path(__file__).resolve().parents[1] / "scripts"


def _load_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    path: Path,
    stubs: dict[str, types.ModuleType],
):
    for k, v in stubs.items():
        monkeypatch.setitem(sys.modules, k, v)
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _make_pil_stub() -> dict[str, types.ModuleType]:
    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")

    class _Opened:
        def convert(self, _mode: str):
            return self

    def open(_path):
        return _Opened()

    pil_image.open = open
    pil.Image = pil_image
    return {"PIL": pil, "PIL.Image": pil_image}


def _make_transformers_stub() -> dict[str, types.ModuleType]:
    tr = types.ModuleType("transformers")

    class BlipProcessor:
        @classmethod
        def from_pretrained(cls, _model_id: str):
            return cls()

        def __call__(self, _image, return_tensors: str = "pt"):
            return {"pixel_values": "dummy"}

        def decode(self, _tokens, skip_special_tokens: bool = True):
            return "a dummy caption"

    class BlipForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, _model_id: str):
            return cls()

        def generate(self, **_inputs):
            return [[0, 1, 2]]

    tr.BlipProcessor = BlipProcessor
    tr.BlipForConditionalGeneration = BlipForConditionalGeneration
    return {"transformers": tr}


def test_caption_images_writes_txt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scripts_dir = _scripts_dir()
    mod = _load_module(
        monkeypatch,
        "caption_images",
        scripts_dir / "caption_images.py",
        {**_make_pil_stub(), **_make_transformers_stub()},
    )

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"")  # existence is enough for iter_images()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "caption_images.py",
            "--images",
            str(images_dir),
            "--prefix",
            "sksSubject",
            "--overwrite",
        ],
    )
    mod.main()

    cap = images_dir / "a.txt"
    assert cap.exists()
    assert cap.read_text(encoding="utf-8") == "sksSubject a dummy caption\n"


def test_iter_images_filters_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scripts_dir = _scripts_dir()
    mod = _load_module(
        monkeypatch,
        "caption_images2",
        scripts_dir / "caption_images.py",
        {**_make_pil_stub(), **_make_transformers_stub()},
    )

    (tmp_path / "x.jpg").write_bytes(b"")
    (tmp_path / "y.txt").write_text("no", encoding="utf-8")
    (tmp_path / "z.webp").write_bytes(b"")

    got = [p.name for p in mod.iter_images(tmp_path)]
    assert got == ["x.jpg", "z.webp"]
