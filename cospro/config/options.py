"""Declare command-line options as dataclass fields and generate argparse parsers from them."""

from __future__ import annotations

import argparse
from dataclasses import MISSING, Field, field, fields
from typing import Any, Callable, Iterable


class ConfigError(ValueError):
    """An invalid configuration; the message is shown to the user unchanged."""


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


# Argument styles:
#   value     --name VALUE
#   flag      --name sets True (store_true)
#   switch    --name, --name true or --name false (optional boolean value)
#   toggle    --name / --no-name (argparse.BooleanOptionalAction)
STYLES = ("value", "flag", "switch", "toggle")


def option(
    default: Any = MISSING,
    *,
    flags: Iterable[str] = (),
    parse: Callable[[str], Any] | None = None,
    style: str = "value",
    choices: Iterable[Any] | None = None,
    help: str | None = None,
    default_factory: Callable[[], Any] | None = None,
) -> Any:
    """A dataclass field that also describes its command-line option."""

    if style not in STYLES:
        raise ValueError(f"Unknown option style {style!r}.")
    metadata = {
        "flags": tuple(flags),
        "parse": parse,
        "style": style,
        "choices": None if choices is None else list(choices),
        "help": help,
    }
    if default_factory is not None:
        return field(default_factory=default_factory, metadata=metadata)
    if style == "flag" and default is MISSING:
        default = False
    return field(default=default, metadata=metadata)


def field_default(item: Field) -> Any:
    if item.default is not MISSING:
        return item.default
    if item.default_factory is not MISSING:
        return item.default_factory()
    raise ValueError(f"Option {item.name} has no default.")


def add_section_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup, section: type) -> None:
    """Add one argument per field of ``section`` to ``parser``."""

    for item in fields(section):
        meta = item.metadata
        flags = meta["flags"] or (f"--{item.name}",)
        kwargs: dict[str, Any] = {"dest": item.name, "default": field_default(item)}
        if meta["help"] is not None:
            kwargs["help"] = meta["help"]
        style = meta["style"]
        if style == "flag":
            kwargs["action"] = "store_true"
        elif style == "toggle":
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            if style == "switch":
                kwargs.update(type=str_to_bool, nargs="?", const=True)
            elif meta["parse"] is not None:
                kwargs["type"] = meta["parse"]
            if meta["choices"] is not None:
                kwargs["choices"] = meta["choices"]
        parser.add_argument(*flags, **kwargs)


def section_defaults(section: type) -> dict[str, Any]:
    return {item.name: field_default(item) for item in fields(section)}


def section_from_namespace(section: type, namespace: argparse.Namespace):
    return section(**{item.name: getattr(namespace, item.name) for item in fields(section)})
