import logging
from pathlib import Path
from typing import Optional

import zmq
import torch
import torch.multiprocessing as mp
from transformers import AutoConfig
from huggingface_hub import snapshot_download
from pymllm.configs import get_global_config
from pymllm.orchestrator.tokenizer_process import TokenizerProcess
from pymllm.orchestrator.detokenizer_process import DetokenizerProcess
from pymllm.orchestrator.model_runner_process import ModelRunnerProcess
from pymllm.orchestrator.async_disk_io_process import AsyncDiskIoProcess
from pymllm.orchestrator.request_response_process import RequestResponseProcess

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self):
        self._config_logging()
        self._set_default_torch_dtype()
        self._check_model_and_tokenizer()

        # Orchestrator, shall we start the music here?
        self._launch_processes()

    def _launch_processes(self):
        """
        TODO issue processes here
        """

        # RR process is the main process
        self._rr_process = RequestResponseProcess()

    def _set_default_torch_dtype(self):
        """Set the default torch dtype based on the server configuration."""
        dtype = get_global_config().server.dtype
        if dtype == "auto":
            dtype = "bfloat16" if torch.cuda.is_available() else "float32"
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype for torch default dtype: {dtype!r}")
        torch.set_default_dtype(torch_dtype)

    def _config_logging(self):
        """Configure logging level from server configuration."""
        level_name = get_global_config().server.log_level.upper()
        level = getattr(logging, level_name, logging.INFO)
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        else:
            root_logger.setLevel(level)
        logging.getLogger("pymllm").setLevel(level)

    def _check_model_and_tokenizer(self):
        cfg = get_global_config()
        if cfg.server.model_path is None or cfg.server.tokenizer_path is None:
            logger.error("Model path or tokenizer path is not set")
            raise ValueError("Model path or tokenizer path is not set")
        model_path = cfg.server.model_path
        tokenizer_path = cfg.server.tokenizer_path
        download_dir = cfg.server.download_dir
        trust_remote_code = cfg.server.trust_remote_code

        shared_path = model_path == tokenizer_path

        model_path = self._maybe_download(model_path, download_dir)
        cfg.server.model_path = model_path

        if shared_path:
            cfg.server.tokenizer_path = model_path
        else:
            cfg.server.tokenizer_path = self._maybe_download(
                tokenizer_path, download_dir
            )

        cfg.model.hf_config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
        )
        logger.info("Loaded model config: %s", cfg.model.hf_config.__class__.__name__)

    @staticmethod
    def _maybe_download(path: Path, download_dir: Optional[Path] = None) -> Path:
        """Return a local directory for *path*, downloading if necessary."""
        if path.is_dir():
            return path

        repo_id = str(path)
        logger.info("Downloading '%s' ...", repo_id)

        kwargs = {}
        if download_dir is not None:
            kwargs["local_dir"] = str(download_dir / path.name)

        downloaded = snapshot_download(repo_id=repo_id, **kwargs)
        logger.info("Downloaded '%s' to '%s'", repo_id, downloaded)
        return Path(downloaded)

    def generate(self, stream: bool = True):
        pass

    async def generate_async(self, stream: bool = True):
        pass
