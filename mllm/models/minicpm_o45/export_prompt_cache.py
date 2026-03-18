#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""Export fixed MiniCPM-o-4_5 token2wav prompt cache for native C++ runtime.

This script extracts prompt_speech_tokens / prompt_mels / speaker_embedding from
one reference wav and writes a compact binary cache.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np


def _setup_cosyvoice2_alias() -> None:
    if "cosyvoice2.flow.flow" in sys.modules:
        return

    import stepaudio2.cosyvoice2.flow.decoder_dit as _step_decoder_dit
    import stepaudio2.cosyvoice2.flow.flow as _step_flow
    import stepaudio2.cosyvoice2.flow.flow_matching as _step_flow_matching
    import stepaudio2.cosyvoice2.transformer.upsample_encoder_v2 as _step_upsample

    cosyvoice2_pkg = types.ModuleType("cosyvoice2")
    cosyvoice2_flow_pkg = types.ModuleType("cosyvoice2.flow")
    cosyvoice2_transformer_pkg = types.ModuleType("cosyvoice2.transformer")

    cosyvoice2_flow_pkg.flow = _step_flow
    cosyvoice2_flow_pkg.flow_matching = _step_flow_matching
    cosyvoice2_flow_pkg.decoder_dit = _step_decoder_dit
    cosyvoice2_transformer_pkg.upsample_encoder_v2 = _step_upsample

    cosyvoice2_pkg.flow = cosyvoice2_flow_pkg
    cosyvoice2_pkg.transformer = cosyvoice2_transformer_pkg

    sys.modules["cosyvoice2"] = cosyvoice2_pkg
    sys.modules["cosyvoice2.flow"] = cosyvoice2_flow_pkg
    sys.modules["cosyvoice2.flow.flow"] = _step_flow
    sys.modules["cosyvoice2.flow.flow_matching"] = _step_flow_matching
    sys.modules["cosyvoice2.flow.decoder_dit"] = _step_decoder_dit
    sys.modules["cosyvoice2.transformer"] = cosyvoice2_transformer_pkg
    sys.modules["cosyvoice2.transformer.upsample_encoder_v2"] = _step_upsample


def _resolve_device(torch_mod, req: str):
    req = req.lower()
    if req == "cuda":
        if not torch_mod.cuda.is_available():
            raise RuntimeError("Requested --device=cuda but CUDA is unavailable")
        return torch_mod.device("cuda")
    if req == "mps":
        if not getattr(torch_mod.backends, "mps", None) or not torch_mod.backends.mps.is_available():
            raise RuntimeError("Requested --device=mps but MPS is unavailable")
        return torch_mod.device("mps")
    if req == "cpu":
        return torch_mod.device("cpu")
    if req != "auto":
        raise ValueError(f"Unsupported --device: {req}")

    if torch_mod.cuda.is_available():
        return torch_mod.device("cuda")
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return torch_mod.device("mps")
    return torch_mod.device("cpu")


def _move_model(model, device):
    if device.type == "cuda" and hasattr(model, "cuda"):
        return model.cuda()
    if device.type == "cpu" and hasattr(model, "cpu"):
        return model.cpu()
    if hasattr(model, "to"):
        return model.to(device)
    return model


class _StageLogger:
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        if not self.verbose:
            return
        dt = time.time() - self.t0
        print(f"[prompt-cache +{dt:.3f}s] {msg}", flush=True)


def _write_cache(path: Path, prompt_tokens: np.ndarray, prompt_mels: np.ndarray, spk_emb: np.ndarray) -> None:
    # File layout (little-endian):
    # magic[8] = "M45PC1\\0\\0"
    # u32 version = 1
    # i32 prompt_token_len
    # i32 prompt_mel_frames
    # i32 mel_dim
    # i32 spk_dim
    # i32[prompt_token_len]
    # f32[prompt_mel_frames * mel_dim]
    # f32[spk_dim]
    magic = b"M45PC1\0\0"
    version = 1
    token_len = int(prompt_tokens.shape[0])
    mel_frames = int(prompt_mels.shape[0])
    mel_dim = int(prompt_mels.shape[1])
    spk_dim = int(spk_emb.shape[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<iiii", token_len, mel_frames, mel_dim, spk_dim))
        f.write(prompt_tokens.astype(np.int32, copy=False).tobytes(order="C"))
        f.write(prompt_mels.astype(np.float32, copy=False).tobytes(order="C"))
        f.write(spk_emb.astype(np.float32, copy=False).tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MiniCPM-o-4_5 fixed prompt cache for native C++ token2wav")
    parser.add_argument("--ref_wav", required=True, help="Reference wav path used for voice style")
    parser.add_argument("--token2wav_dir", required=True, help="Path to assets/token2wav directory")
    parser.add_argument("--python_src_root", required=True, help="Path to MiniCPM-o-4_5 python_src_code directory")
    parser.add_argument("--out_cache", required=True, help="Output cache path (.bin)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Runtime device")
    parser.add_argument("--verbose", action="store_true", help="Print detailed stage logs")
    args = parser.parse_args()

    python_src_root = Path(args.python_src_root).expanduser().resolve()
    ref_wav = Path(args.ref_wav).expanduser().resolve()
    token2wav_dir = Path(args.token2wav_dir).expanduser().resolve()
    out_cache = Path(args.out_cache).expanduser().resolve()

    if str(python_src_root) not in sys.path:
        sys.path.insert(0, str(python_src_root))

    import onnxruntime
    import s3tokenizer
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    from hyperpyyaml import load_hyperpyyaml
    from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

    logger = _StageLogger(args.verbose)

    logger.log("Resolving runtime device...")
    device = _resolve_device(torch, args.device)
    print(f"[prompt-cache] device={device}", flush=True)
    print(f"[prompt-cache] ref_wav={ref_wav}", flush=True)
    print(f"[prompt-cache] token2wav_dir={token2wav_dir}", flush=True)

    _setup_cosyvoice2_alias()

    logger.log("Loading speech tokenizer ONNX...")
    audio_tokenizer = s3tokenizer.load_model(str(token2wav_dir / "speech_tokenizer_v2_25hz.onnx"))
    audio_tokenizer = _move_model(audio_tokenizer, device)
    if hasattr(audio_tokenizer, "eval"):
        audio_tokenizer = audio_tokenizer.eval()

    logger.log("Loading campplus.onnx...")
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    spk_model = onnxruntime.InferenceSession(
        str(token2wav_dir / "campplus.onnx"),
        sess_options=option,
        providers=["CPUExecutionProvider"],
    )

    logger.log("Reading flow.yaml for up_rate...")
    with open(token2wav_dir / "flow.yaml", "r", encoding="utf-8") as f:
        cfg = load_hyperpyyaml(f)
    up_rate = int(cfg["flow"].up_rate)
    print(f"[prompt-cache] flow.up_rate={up_rate}", flush=True)

    logger.log("Preparing prompt speech tokens (16k)...")
    audio = s3tokenizer.load_audio(str(ref_wav), sr=16000)
    mels = s3tokenizer.log_mel_spectrogram(audio)
    mels, mels_lens = s3tokenizer.padding([mels])

    quantize_device = device
    try:
        prompt_tokens, prompt_tokens_lens = audio_tokenizer.quantize(mels.to(quantize_device), mels_lens.to(quantize_device))
    except Exception:
        quantize_device = torch.device("cpu")
        audio_tokenizer = _move_model(audio_tokenizer, quantize_device)
        if hasattr(audio_tokenizer, "eval"):
            audio_tokenizer = audio_tokenizer.eval()
        prompt_tokens, prompt_tokens_lens = audio_tokenizer.quantize(mels.to(quantize_device), mels_lens.to(quantize_device))

    prompt_tokens = prompt_tokens.to(device)
    prompt_tokens_lens = prompt_tokens_lens.to(device)
    logger.log(f"prompt_tokens shape={tuple(prompt_tokens.shape)}, lens={prompt_tokens_lens.tolist()}")

    logger.log("Preparing speaker embedding...")
    spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
    spk_emb_np = spk_model.run(
        None,
        {spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()},
    )[0]
    spk_emb = torch.tensor(spk_emb_np, device=device, dtype=torch.float32)
    logger.log(f"spk_emb shape={tuple(spk_emb.shape)}")

    logger.log("Preparing prompt mel (24k)...")
    audio_24k, sample_rate = torchaudio.load(str(ref_wav), backend="soundfile")
    audio_24k = audio_24k.mean(dim=0, keepdim=True)
    if sample_rate != 24000:
        audio_24k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio_24k)
    prompt_mel = mel_spectrogram(audio_24k).transpose(1, 2).squeeze(0)  # [T, 80]
    prompt_mels = prompt_mel.unsqueeze(0).to(device)
    target_len = int(prompt_tokens.shape[1]) * up_rate
    if target_len > prompt_mels.shape[1]:
        prompt_mels = torch.nn.functional.pad(
            prompt_mels,
            (0, 0, 0, target_len - prompt_mels.shape[1]),
            mode="replicate",
        )
    logger.log(f"prompt_mels shape={tuple(prompt_mels.shape)}")

    logger.log("Writing cache...")
    token_np = prompt_tokens[0].detach().cpu().numpy().astype(np.int32)
    mel_np = prompt_mels[0].detach().cpu().numpy().astype(np.float32)
    spk_np = spk_emb[0].detach().cpu().numpy().astype(np.float32)
    _write_cache(out_cache, token_np, mel_np, spk_np)

    print(f"[prompt-cache] wrote: {out_cache}", flush=True)
    print(f"[prompt-cache] token_len={token_np.shape[0]}, mel_frames={mel_np.shape[0]}, mel_dim={mel_np.shape[1]}, spk_dim={spk_np.shape[0]}",
          flush=True)


if __name__ == "__main__":
    main()

