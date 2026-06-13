import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from pymllm.mobile.backends.qualcomm.transformers.core.qdq import (
    ActivationQDQ,
    FixedActivationQDQ,
)
from pymllm.mobile.backends.qualcomm.transformers.core.rms_norm import QRMSNorm
from pymllm.mobile.backends.qualcomm.transformers.core.qlinear import (
    QLinearLPBQ,
    QLinearW8A16_PerChannelSym,
)
from pymllm.mobile.backends.qualcomm.transformers.core.embedding import QEmbedding
from pymllm.mobile.backends.qualcomm.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from pymllm.mobile.backends.qualcomm.transformers.core.observer import ConcatObserver


def recompute_scale_zp(module):
    """
    Callback function: Forcefully refresh scale and zero_point of all FakeQuantize modules after calibration.

    When using ConcatObserver, min/max may be updated during forward pass,
    but scale/zp stored in FakeQuantize's internal buffer may still be from old min/max.
    This forces a calculate_qparams call to sync the latest parameters.
    """
    if isinstance(module, ActivationQDQ):
        observer = module.fake_quant.activation_post_process

        if hasattr(observer, "min_val") and hasattr(observer, "max_val"):
            if observer.min_val.numel() == 0 or observer.max_val.numel() == 0:
                return
            if (
                torch.isinf(observer.min_val).any()
                or torch.isinf(observer.max_val).any()
            ):
                return

            try:
                scale, zero_point = observer.calculate_qparams()
            except Exception as e:
                print(e)
                return

            if (
                hasattr(module.fake_quant, "scale")
                and module.fake_quant.scale is not None
            ):
                if module.fake_quant.scale.shape != scale.shape:
                    module.fake_quant.scale.resize_(scale.shape)
                module.fake_quant.scale.copy_(scale)

            if (
                hasattr(module.fake_quant, "zero_point")
                and module.fake_quant.zero_point is not None
            ):
                if module.fake_quant.zero_point.shape != zero_point.shape:
                    module.fake_quant.zero_point.resize_(zero_point.shape)
                module.fake_quant.zero_point.copy_(zero_point)


def validate_concat_observer_fn(module, results: list, name: str = ""):
    """Validate that all input_observers in ConcatObserver have consistent scale and zero_point."""
    if not isinstance(module, ConcatObserver):
        return

    input_observers = module.input_observers
    if len(input_observers) == 0:
        return

    scales_zps = []
    for i, observer in enumerate(input_observers):
        try:
            scale, zp = observer.calculate_qparams()
            scales_zps.append(f"[{i}] s={scale.item():.8f} zp={zp.item()}")
        except Exception:
            scales_zps.append(f"[{i}] failed")

    print(f"ConcatObserver [{name}]: {' | '.join(scales_zps)}")

    if len(input_observers) <= 1:
        return

    first_observer = input_observers[0]
    try:
        ref_scale, ref_zp = first_observer.calculate_qparams()
    except Exception:
        return

    for i, observer in enumerate(input_observers[1:], start=1):
        try:
            scale, zp = observer.calculate_qparams()
        except Exception:
            results.append(f"Failed to calculate qparams for observer[{i}]")
            continue

        scale_match = torch.allclose(ref_scale, scale, rtol=1e-5, atol=1e-8)
        zp_match = torch.equal(ref_zp, zp)

        if not scale_match or not zp_match:
            results.append(
                f"observer[{i}] mismatch: ref_scale={ref_scale.item():.8f}, "
                f"scale={scale.item():.8f}, ref_zp={ref_zp.item()}, zp={zp.item()}"
            )


def freeze_rmsnorm_weight(m):
    if isinstance(m, QRMSNorm):
        m.freeze_weight()


def freeze_linear_weight(m):
    if isinstance(m, QLinearLPBQ) or isinstance(m, QLinearW8A16_PerChannelSym):
        m.freeze_weight()


def freeze_embed_tokens_weight(m):
    if isinstance(m, QEmbedding):
        m.freeze_weight()


def disable_qdq_observer(m):
    if isinstance(m, ActivationQDQ):
        m.disable_observer()


def enable_qdq_observer(m):
    if isinstance(m, ActivationQDQ):
        m.enable_observer()


def enable_fake_quant(m):
    if isinstance(m, ActivationQDQ) or isinstance(m, FixedActivationQDQ):
        m.enable_fakequant()
    if isinstance(m, QLinearLPBQ):
        m.enable_fakequant()
    if isinstance(m, QRMSNorm):
        m.enable_fakequant()
    if isinstance(m, QEmbedding):
        m.enable_fakequant()


def disable_fake_quant(m):
    if isinstance(m, ActivationQDQ) or isinstance(m, FixedActivationQDQ):
        m.disable_fakequant()
    if isinstance(m, QLinearLPBQ):
        m.disable_fakequant()
    if isinstance(m, QRMSNorm):
        m.disable_fakequant()
    if isinstance(m, QEmbedding):
        m.disable_fakequant()


def convert_weight(m):
    if isinstance(m, QLinearLPBQ) or isinstance(m, QLinearW8A16_PerChannelSym):
        m.convert_to_conv2d_deploy_hwio()
    if isinstance(m, QRMSNorm):
        m.convert_to_deploy()
    if isinstance(m, QEmbedding):
        m.convert_to_deploy()


class Qwen3_5Quantizer:
    def __init__(self, model_path: str, mllm_qualcomm_max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            dtype=torch.float32,
        )
        self.model.cuda()
        self.mllm_qualcomm_max_length = mllm_qualcomm_max_length
        self.model.mllm_qualcomm_max_length = mllm_qualcomm_max_length

        if self.model.config.tie_word_embeddings:
            self.model.copy_lm_head_weight_from_embed_tokens()

        # PTQ All Weights (only affects quantized modules — GDN layers are skipped)
        self.model.apply(freeze_rmsnorm_weight)
        self.model.apply(freeze_linear_weight)
        self.model.apply(freeze_embed_tokens_weight)
        print("All PTQ weights preparation done.")

    def freeze_activation(self):
        self.model.apply(disable_qdq_observer)

    def enable_activation_update(self):
        self.model.apply(enable_qdq_observer)

    def enable_fake_quant(self):
        self.model.apply(enable_fake_quant)

    def disable_fake_quant(self):
        self.model.apply(disable_fake_quant)

    def compile(self):
        print("Compile Start.")
        self.model = torch.compile(
            self.model, mode="reduce-overhead", fullgraph=False, backend="inductor"
        )
        print("Compile done.")

    def infer(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.mllm_qualcomm_max_length
            - len(model_inputs.input_ids[0])
            - 1,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print("content:", content)

    def calibrate(self, num_samples=64, max_seq_length=512):
        """
        Perform calibration using Wikipedia dataset (PTQ).
        Only full attention layers (with QDQ nodes) are affected;
        GDN layers pass through without quantization.
        """
        print(
            f"Starting calibration, samples: {num_samples}, max length: {max_seq_length}"
        )

        self.enable_activation_update()
        self.model.eval()

        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-v1",
            split="train",
            streaming=True,
        )

        samples_processed = 0

        with torch.no_grad():
            pbar = tqdm(total=num_samples, desc="Calibrating")
            for entry in dataset:
                if samples_processed >= num_samples:
                    break

                if len(entry["text"].strip()) < 1024:
                    continue

                messages = [{"role": "user", "content": entry["text"]}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = self.tokenizer(
                    [text],
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                    padding=False,
                ).to(self.model.device)

                self.model.generate(
                    **model_inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

                samples_processed += 1
                pbar.update(1)

        self.freeze_activation()
        print("\nCalibration completed, activation quantization parameters frozen.")

    def convert(self):
        self.model.apply(convert_weight)
        self.model.model.convert_rope_for_deploy()

    def recompute_scale_zp(self):
        self.model.apply(recompute_scale_zp)

    def validate_concat_observer(self):
        results = []
        for name, module in self.model.named_modules():
            validate_concat_observer_fn(module, results, name)
        if results:
            print("ConcatObserver validation FAILED:")
            for msg in results:
                print(f"  {msg}")
            raise ValueError("ConcatObserver validation FAILED")
        else:
            print(
                "ConcatObserver validation PASSED: all observers have matching scale and zp"
            )
        print("ConcatObserver validation done.", flush=True)
