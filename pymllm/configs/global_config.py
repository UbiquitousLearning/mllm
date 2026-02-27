"""Global configuration singleton aggregating all sub-configs."""

from __future__ import annotations

import argparse
import types
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pymllm.configs.server_config import ServerConfig
from pymllm.configs.model_config import ModelConfig
from pymllm.configs.quantization_config import QuantizationConfig


@dataclass
class GlobalConfig:
    """Singleton that holds every sub-config pymllm needs.

    Usage::

        from pymllm.configs import get_global_config

        cfg = get_global_config()
        cfg.model.model_path
        cfg.model.hidden_size
        cfg.quantization.method
        cfg.server.host
    """

    server: "ServerConfig" = field(default=None, repr=False)  # type: ignore[assignment]
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    _initialized: bool = field(default=False, repr=False)

    def __new__(cls):
        if not hasattr(cls, "_instance") or cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __post_init__(self):
        if self.server is None:
            self.server = ServerConfig(model_path=None)

    @classmethod
    def get_instance(cls) -> "GlobalConfig":
        if not hasattr(cls, "_instance") or cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (useful in tests)."""
        cls._instance = None


def _parse_bool(value: Any) -> bool:
    """Convert common CLI boolean spellings into ``bool``.

    This helper is intentionally permissive because CLI users often provide
    booleans in different forms (for example ``true``, ``1``, ``yes``,
    ``false``, ``0``, ``no``). The function raises ``argparse.ArgumentTypeError``
    to integrate naturally with ``argparse`` validation and error reporting.
    """

    if isinstance(value, bool):
        return value
    if value is None:
        return True

    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Expected one of true/false, 1/0, yes/no."
    )


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner_type, is_optional)`` for Optional/Union annotations."""

    origin = get_origin(annotation)
    if origin not in (Union, types.UnionType):
        return annotation, False

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1 and len(get_args(annotation)) == 2:
        return args[0], True
    return annotation, False


def _converter_for_annotation(annotation: Any) -> Optional[Callable[[str], Any]]:
    """Map a type annotation to an ``argparse`` converter.

    Only scalar, CLI-friendly annotations are supported. Complex runtime fields
    (for example nested dict/object handles) are intentionally excluded from the
    generated CLI surface to keep the interface predictable and safe.
    """

    inner, _ = _unwrap_optional(annotation)
    origin = get_origin(inner)
    if origin is not None:
        if origin is Literal:
            literal_values = get_args(inner)
            if literal_values:
                return type(literal_values[0])
            return str
        return None

    if inner in (str, int, float):
        return inner
    if inner is Path:
        return Path
    return None


def _is_bool_annotation(annotation: Any) -> bool:
    """Return ``True`` if annotation represents a bool/Optional[bool] field."""

    inner, _ = _unwrap_optional(annotation)
    return inner is bool


def _format_default_for_help(value: Any) -> str:
    """Create a concise, readable default string for CLI help text."""

    if value is MISSING:
        return "<required>"
    if value is None:
        return "None"
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def make_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Create an ``argparse`` parser with two-level GlobalConfig CLI options.

    The generated options follow the naming pattern ``--<section>.<field>`` so
    each sub-config can be configured independently:

    - ``server`` options map to :class:`ServerConfig` fields.
    - ``model`` options map to :class:`ModelConfig` fields.
    - ``quantization`` options map to :class:`QuantizationConfig` fields.

    Examples
    --------
    - ``--server.host 0.0.0.0``
    - ``--server.port 8080``
    - ``--server.sleep_on_idle`` (implicit true)
    - ``--server.sleep_on_idle false`` (explicit false)
    - ``--quantization.method awq``

    Design notes
    ------------
    - Options are generated from dataclass metadata, which keeps the CLI surface
      synchronized with config definitions and avoids manual drift.
    - Parser defaults are suppressed (``argparse.SUPPRESS``), so ``read_args``
      can reliably detect whether a value was explicitly provided by the user.
    - Only CLI-friendly scalar fields are exposed; runtime-only fields are
      skipped automatically.
    """

    if parser is None:
        parser = argparse.ArgumentParser(
            prog="pymllm",
            description="CLI options for configuring pymllm GlobalConfig.",
        )

    cfg = GlobalConfig.get_instance()
    sections: list[tuple[str, Any]] = [
        ("server", cfg.server),
        ("model", cfg.model),
        ("quantization", cfg.quantization),
    ]

    for section_name, section_obj in sections:
        section_group = parser.add_argument_group(
            f"{section_name} config",
            f"Options for the '{section_name}' section of GlobalConfig.",
        )
        type_hints = get_type_hints(type(section_obj))
        for dc_field in fields(section_obj):
            if dc_field.name.startswith("_"):
                continue

            annotation = type_hints.get(dc_field.name, dc_field.type)
            option = f"--{section_name}.{dc_field.name}"
            dest = f"{section_name}__{dc_field.name}"
            default_value = getattr(section_obj, dc_field.name)

            if _is_bool_annotation(annotation):
                section_group.add_argument(
                    option,
                    dest=dest,
                    nargs="?",
                    const=True,
                    type=_parse_bool,
                    default=argparse.SUPPRESS,
                    help=(
                        f"{section_name}.{dc_field.name} (bool, default: "
                        f"{_format_default_for_help(default_value)}). "
                        "Can be provided as a flag for true or with an explicit value."
                    ),
                )
                continue

            converter = _converter_for_annotation(annotation)
            if converter is None:
                # Skip non-scalar or runtime-only fields (e.g. arbitrary objects).
                continue

            section_group.add_argument(
                option,
                dest=dest,
                type=converter,
                default=argparse.SUPPRESS,
                help=(
                    f"{section_name}.{dc_field.name} (default: "
                    f"{_format_default_for_help(default_value)})."
                ),
            )

    return parser


def read_args(
    argv: Optional[Sequence[str]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
) -> GlobalConfig:
    """Parse CLI args and apply overrides to the singleton ``GlobalConfig``.

    Parameters
    ----------
    argv
        Optional argument vector. If ``None``, ``argparse`` reads from
        ``sys.argv`` (standard CLI behavior).
    parser
        Optional parser to use. When omitted, this function builds one through
        :func:`make_args`.

    Returns
    -------
    GlobalConfig
        The singleton config instance after CLI overrides have been applied.

    Behavior
    --------
    1. Parse all generated ``--section.field`` options.
    2. Apply only explicitly provided options (no accidental overwrite by parser
       defaults).
    3. Rebuild ``ServerConfig`` when server fields change so validation in
       ``ServerConfig.__post_init__`` and ``_validate`` remains enforced.
    4. Keep ``server.model_path`` and ``model.model_path`` aligned when only one
       side is explicitly overridden (the same precedence used by runtime config
       loading conventions).
    """

    if parser is None:
        parser = make_args()

    namespace = parser.parse_args(argv)
    parsed = vars(namespace)
    cfg = GlobalConfig.get_instance()

    # Server: reconstruct to preserve validation behavior.
    from pymllm.configs.server_config import ServerConfig

    server_updates: dict[str, Any] = {}
    for dc_field in fields(cfg.server):
        key = f"server__{dc_field.name}"
        if key in parsed:
            server_updates[dc_field.name] = parsed[key]
    if server_updates:
        server_values = {
            dc_field.name: getattr(cfg.server, dc_field.name)
            for dc_field in fields(cfg.server)
        }
        server_values.update(server_updates)
        cfg.server = ServerConfig(**server_values)

    # Model / Quantization: in-place updates are sufficient.
    for section_name, section_obj in (
        ("model", cfg.model),
        ("quantization", cfg.quantization),
    ):
        for dc_field in fields(section_obj):
            key = f"{section_name}__{dc_field.name}"
            if key in parsed:
                setattr(section_obj, dc_field.name, parsed[key])

    # Keep model path synchronized when only one side is explicitly overridden.
    server_model_overridden = "server__model_path" in parsed
    model_model_overridden = "model__model_path" in parsed
    if server_model_overridden and not model_model_overridden:
        cfg.model.model_path = cfg.server.model_path
    elif model_model_overridden and not server_model_overridden:
        cfg.server.model_path = cfg.model.model_path

    cfg._initialized = True
    return cfg


def get_global_config() -> GlobalConfig:
    """Return the global config singleton."""
    return GlobalConfig.get_instance()
