import json
from typing import List, Optional
from typing import Any, Dict, Tuple, get_origin, get_args


def _safe_import_peft():
    try:
        import peft  # type: ignore
        from peft.config import PeftConfig  # type: ignore
        return peft, PeftConfig
    except Exception:
        return None, None


def list_peft_tuners() -> List[str]:
    """Return available PEFT tuner names discovered from installed peft.

    Names are class names without the trailing "Config" (e.g., "Lora", "AdaLora", "IA3").
    """
    peft, PeftConfig = _safe_import_peft()
    if peft is None or PeftConfig is None:
        return []

    tuners: List[str] = []
    for name in dir(peft):
        attr = getattr(peft, name)
        try:
            if isinstance(attr, type) and issubclass(attr, PeftConfig) and attr is not PeftConfig:
                if name.endswith("Config"):
                    tuners.append(name[: -len("Config")])
        except Exception:
            continue

    # Deduplicate and sort
    tuners = sorted(set(tuners))
    return tuners


def default_peft_config_json(tuner_name: str, include_optionals: bool = True) -> str:
    """Create a default JSON string for the given tuner config using its defaults.

    Falls back to an empty JSON object string if unavailable.
    """
    peft, _ = _safe_import_peft()
    if peft is None:
        return "{}"

    cfg_cls_name = f"{tuner_name}Config" if not tuner_name.endswith("Config") else tuner_name
    cfg_cls = getattr(peft, cfg_cls_name, None)
    if cfg_cls is None:
        return "{}"

    # Try to instantiate with defaults
    try:
        cfg = cfg_cls()
        # Dataclass to dict: prefer .to_dict if provided
        data = None
        if hasattr(cfg, "to_dict"):
            try:
                data = cfg.to_dict()
            except Exception:
                data = None
        if data is None:
            try:
                from dataclasses import asdict, is_dataclass, fields

                if is_dataclass(cfg):
                    data = asdict(cfg)
                    if include_optionals:
                        # Ensure all annotated fields exist (even if None)
                        for f in fields(cfg):
                            data.setdefault(f.name, getattr(cfg, f.name, None))
                else:
                    data = dict(cfg.__dict__)
            except Exception:
                data = dict(cfg.__dict__)
    except Exception:
        data = {}

    # For convenience, ensure target_modules key exists
    # Ensure presence of commonly-used fields for UX
    if "target_modules" not in data:
        data["target_modules"] = []
    if "modules_to_save" not in data:
        data["modules_to_save"] = []

    # Remove metadata/noise fields that are auto-determined by PEFT and not user-tunable
    # to keep the JSON editor focused.
    for k in ("peft_type", "peft_version"):
        try:
            data.pop(k)
        except Exception:
            pass

    # Compact keys unless user wants optionals
    if not include_optionals:
        data = {k: v for k, v in data.items() if v is not None}
    return json.dumps(data, indent=2, sort_keys=True)


def peft_field_docs(tuner_name: str) -> List[Tuple[str, Optional[str]]]:
    """Return [(field_name, help_text or None), ...] using dataclass metadata."""
    peft, _ = _safe_import_peft()
    if peft is None:
        return []
    cfg_cls_name = f"{tuner_name}Config" if not tuner_name.endswith("Config") else tuner_name
    cfg_cls = getattr(peft, cfg_cls_name, None)
    if cfg_cls is None:
        return []
    try:
        from dataclasses import fields, is_dataclass

        docs: List[Tuple[str, Optional[str]]] = []
        if is_dataclass(cfg_cls):
            for f in fields(cfg_cls):
                help_text = None
                try:
                    md = f.metadata or {}
                    help_text = md.get("help")
                except Exception:
                    help_text = None
                docs.append((f.name, help_text))
            return docs
    except Exception:
        pass
    # Fallback: no dataclass metadata available
    try:
        anns = getattr(cfg_cls, "__annotations__", {})
        return [(k, None) for k in anns.keys()]
    except Exception:
        return []


def peft_help_markdown(tuner_name: str) -> str:
    """Build a Markdown help list from PEFT config metadata.

    We rely on the Markdown component's `max_height` parameter for scrolling.
    """
    docs = peft_field_docs(tuner_name)
    if not docs:
        return ""
    lines = [f"### {tuner_name} parameters", ""]
    for name, help_text in docs:
        if help_text:
            lines.append(f"- `{name}`: {help_text}")
        else:
            lines.append(f"- `{name}`")
    return "\n".join(lines)


def get_tuner_schema(tuner_name: str) -> list[Dict[str, Any]]:
    """Return a schema for the tuner config fields.

    Each item: {
      'name': str,
      'category': 'bool'|'int'|'float'|'str'|'enum'|'list_str'|'list_int'|'list_float'|'dict',
      'choices': Optional[List[Any]],
      'default': Any,
      'help': Optional[str],
    }
    """
    peft, _ = _safe_import_peft()
    if peft is None:
        return []
    cfg_cls_name = f"{tuner_name}Config" if not tuner_name.endswith("Config") else tuner_name
    cfg_cls = getattr(peft, cfg_cls_name, None)
    if cfg_cls is None:
        return []

    try:
        from dataclasses import fields, is_dataclass
        import typing, types as pytypes

        # Instantiate defaults
        try:
            cfg = cfg_cls()
            defaults = getattr(cfg, "__dict__", {})
        except Exception:
            cfg = None
            defaults = {}

        schema: list[Dict[str, Any]] = []
        if is_dataclass(cfg_cls):
            # Resolve postponed annotations to concrete types
            try:
                type_hints = typing.get_type_hints(cfg_cls)
            except Exception:
                type_hints = {}
            for f in fields(cfg_cls):
                name = f.name
                ann = type_hints.get(name, f.type)
                help_text = None
                try:
                    help_text = (f.metadata or {}).get("help")
                except Exception:
                    pass

                cat = "str"
                choices = None

                def unwrap_optional(t):
                    o = get_origin(t)
                    if o in (typing.Union, getattr(pytypes, 'UnionType', None)):
                        args = [a for a in get_args(t) if a is not type(None)]
                        return args[0] if args else t
                    return t

                base = unwrap_optional(ann)
                o = get_origin(base)
                a = get_args(base)

                if base is bool:
                    cat = "bool"
                elif base is int:
                    cat = "int"
                elif base is float:
                    cat = "float"
                elif base is str:
                    cat = "str"
                elif str(base).startswith("typing.Literal") or o is typing.Literal:  # type: ignore[attr-defined]
                    cat = "enum"
                    choices = list(a)
                elif o in (list, tuple):
                    subtype = a[0] if a else str
                    if subtype in (int,):
                        cat = "list_int"
                    elif subtype in (float,):
                        cat = "list_float"
                    else:
                        cat = "list_str"
                elif base in (dict, Dict):
                    cat = "dict"
                else:
                    # fallback
                    cat = "str"

                default = defaults.get(name, None)
                schema.append({
                    "name": name,
                    "category": cat,
                    "choices": choices,
                    "default": default,
                    "help": help_text,
                })
        return schema
    except Exception:
        return []
