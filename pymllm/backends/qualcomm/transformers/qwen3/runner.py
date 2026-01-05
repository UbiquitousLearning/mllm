import torch
from tqdm import tqdm
from modelscope.msdatasets import MsDataset
from transformers import AutoTokenizer
from pymllm.backends.qualcomm.transformers.core.qdq import ActivationQDQ
from pymllm.backends.qualcomm.transformers.core.rms_norm import QRMSNorm
from pymllm.backends.qualcomm.transformers.core.qlinear import (
    QLinearLPBQ,
    QLinearW8A16_PerChannelSym,
)
from pymllm.backends.qualcomm.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def freeze_qwen3_rmsnorm_weight(m):
    if isinstance(m, QRMSNorm):
        m.freeze_weight()


def freeze_qwen3_linear_weight(m):
    if isinstance(m, QLinearLPBQ) or isinstance(m, QLinearW8A16_PerChannelSym):
        m.freeze_weight()


def disable_qdq_observer(m):
    if isinstance(m, ActivationQDQ):
        m.disable_observer()


def enable_qdq_observer(m):
    if isinstance(m, ActivationQDQ):
        m.enable_observer()


def convert_weight(m):
    if (
        isinstance(m, QLinearLPBQ)
        or isinstance(m, QLinearW8A16_PerChannelSym)
        or isinstance(m, QRMSNorm)
    ):
        m.convert_to_deploy()


class Qwen3Quantizer:
    def __init__(self, model_path: str, mllm_qualcomm_max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
        )
        self.mllm_qualcomm_max_length = mllm_qualcomm_max_length
        self.model.mllm_qualcomm_max_length = mllm_qualcomm_max_length

        # PTQ All Weights.
        self.model.apply(freeze_qwen3_rmsnorm_weight)
        self.model.apply(freeze_qwen3_linear_weight)
        print("All PTQ weights preparation done.")

    def freeze_activation(self):
        self.model.apply(disable_qdq_observer)

    def enable_activation_update(self):
        self.model.apply(enable_qdq_observer)

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
            enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
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
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)

    def calibrate(self, num_samples=64, max_seq_length=512):
        """
        Perform calibration using Wikipedia dataset (PTQ)
        :param num_samples: Number of samples for calibration
        :param max_seq_length: Maximum length for each sample (not exceeding mllm_qualcomm_max_length)
        """
        print(
            f"Starting calibration, samples: {num_samples}, max length: {max_seq_length}"
        )

        # 1. Enable QDQ Observer for activation values
        self.enable_activation_update()
        self.model.eval()

        # 2. Load Wikipedia dataset (English version example)
        # Use streaming=True to download and process on the fly, without downloading the full几十G dataset
        dataset = MsDataset.load(
            "modelscope/wikitext",
            subset_name="wikitext-103-v1",
            split="train",
            trust_remote_code=True,
        )

        # 3. Execute forward pass (Prefill stage)
        samples_processed = 0

        # Ensure no gradient calculation during inference
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
                    enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
                )
                model_inputs = self.tokenizer(
                    [text],
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                    padding=False,
                ).to(self.model.device)

                # Only need Prefill stage: directly call forward
                # This will trigger observer update statistics in ActivationQDQ
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

        # 4. Close Observer, freeze calibrated quantization parameters
        self.freeze_activation()
        print("\nCalibration completed, activation quantization parameters frozen.")

    def convert(self):
        self.model.apply(convert_weight)
        self.model.model.convert_rope_for_deploy()
