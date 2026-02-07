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


def _make_torch_stub(cuda_available: bool) -> types.ModuleType:
    t = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

    class Generator:
        def __init__(self, device: str = "cpu"):
            self.device = device
            self.seed = None

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    t.cuda = _Cuda()
    t.float16 = object()
    t.float32 = object()
    t.Generator = Generator
    t.dtype = object
    return t


def _make_diffusers_stub(raise_attention: bool) -> types.ModuleType:
    d = types.ModuleType("diffusers")

    class FakeImage:
        def save(self, path):
            Path(path).write_bytes(b"fake")

    class FakeResult:
        def __init__(self):
            self.images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.to_calls: list[str] = []
            self.lora_calls: list[tuple[str, str | None]] = []
            self.calls: list[dict] = []
            self.raise_attention = raise_attention
            self.attention_slicing_called = 0
            self.cpu_offload_called = 0

        def to(self, device: str):
            self.to_calls.append(device)
            return self

        def load_lora_weights(self, parent: str, weight_name: str | None = None):
            self.lora_calls.append((parent, weight_name))

        def enable_attention_slicing(self):
            self.attention_slicing_called += 1
            if self.raise_attention:
                raise RuntimeError("boom")

        def enable_model_cpu_offload(self):
            self.cpu_offload_called += 1

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return FakeResult()

    class StableDiffusionXLPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return FakePipe()

        @classmethod
        def from_single_file(cls, *_args, **_kwargs):
            return FakePipe()

    d.StableDiffusionXLPipeline = StableDiffusionXLPipeline
    return d


def test_infer_main_cpu_dir_seed_increments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scripts_dir = _scripts_dir()
    torch_stub = _make_torch_stub(cuda_available=False)
    diffusers_stub = _make_diffusers_stub(raise_attention=False)

    mod = _load_module(
        monkeypatch,
        "infer_sdxl_lora",
        scripts_dir / "infer_sdxl_lora.py",
        {"torch": torch_stub, "diffusers": diffusers_stub},
    )

    base_dir = tmp_path / "base_dir"
    base_dir.mkdir()
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(mod.time, "strftime", lambda _fmt: "20260101_000000")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_sdxl_lora.py",
            "--base-model",
            str(base_dir),
            "--lora",
            str(lora),
            "--prompt",
            "hello",
            "--negative-prompt",
            "bad",
            "--out-dir",
            str(out_dir),
            "--num-images",
            "2",
            "--seed",
            "10",
            "--device",
            "cpu",
            "--attention-slicing",
            "--cpu-offload",
        ],
    )
    mod.main()

    p0 = out_dir / "sdxl_lora_20260101_000000_00_seed10.png"
    p1 = out_dir / "sdxl_lora_20260101_000000_01_seed11.png"
    assert p0.exists()
    assert p1.exists()


def test_infer_main_cuda_file_random_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scripts_dir = _scripts_dir()
    torch_stub = _make_torch_stub(cuda_available=True)
    diffusers_stub = _make_diffusers_stub(raise_attention=True)  # exercise try/except

    mod = _load_module(
        monkeypatch,
        "infer_sdxl_lora2",
        scripts_dir / "infer_sdxl_lora.py",
        {"torch": torch_stub, "diffusers": diffusers_stub},
    )

    base_file = tmp_path / "base.safetensors"
    base_file.write_bytes(b"")
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(mod.time, "strftime", lambda _fmt: "20260101_000000")
    monkeypatch.setattr(mod.random, "randint", lambda _a, _b: 777)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_sdxl_lora.py",
            "--base-model",
            str(base_file),
            "--lora",
            str(lora),
            "--prompt",
            "hello",
            "--negative-prompt",
            "",
            "--out-dir",
            str(out_dir),
            "--num-images",
            "1",
            "--seed",
            "-1",
            "--attention-slicing",
        ],
    )
    mod.main()

    p0 = out_dir / "sdxl_lora_20260101_000000_00_seed777.png"
    assert p0.exists()


def test_load_pipe_errors_when_no_from_single_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scripts_dir = _scripts_dir()
    torch_stub = _make_torch_stub(cuda_available=False)

    # load module with torch stub only; diffusers is imported lazily inside load_pipe()
    mod = _load_module(
        monkeypatch,
        "infer_sdxl_lora3",
        scripts_dir / "infer_sdxl_lora.py",
        {"torch": torch_stub},
    )

    base_file = tmp_path / "base.safetensors"
    base_file.write_bytes(b"")

    d = types.ModuleType("diffusers")

    class StableDiffusionXLPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("unexpected")

        # intentionally no from_single_file

    d.StableDiffusionXLPipeline = StableDiffusionXLPipeline
    monkeypatch.setitem(sys.modules, "diffusers", d)

    with pytest.raises(SystemExit) as e:
        mod.load_pipe(str(base_file), "cpu", mod.torch.float32)
    assert "from_single_file" in str(e.value)