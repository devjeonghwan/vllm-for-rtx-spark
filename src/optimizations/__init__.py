"""Small option declarations shared by recipe configuration and application."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable


MISSING = object()


@dataclass(frozen=True)
class Option:
    name: str
    value_type: type
    default: object = MISSING
    apply: Callable | None = None
    validate: Callable | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.name == "optimizations":
            raise ValueError("An option needs a nonempty, unreserved name")
        if not isinstance(self.value_type, type):
            raise TypeError("An option needs a value type")
        if any(callback is not None and not callable(callback) for callback in (self.apply, self.validate)):
            raise TypeError("Option apply and validate hooks must be callable")


MODEL = Option("model", str)


def resolve_options(options, declarations):
    """Resolve defaults and reject misspellings before performing any mutation.

    Flat recipe keys remain supported. ``optimizations`` is an optional grouping
    of the same declared keys, rather than a second source of option behavior.
    """
    if not isinstance(options, dict):
        raise TypeError("additional_config.rtx_spark must be an object")
    supplied = dict(options)
    nested = supplied.pop("optimizations", {})
    if not isinstance(nested, dict):
        raise TypeError("rtx_spark.optimizations must be an object")
    if any(not isinstance(name, str) for name in (*supplied, *nested)):
        raise TypeError("RTX Spark option names must be strings")
    if "model" in nested:
        raise ValueError("rtx_spark.model belongs outside optimizations")
    for name, value in nested.items():
        if name in supplied and (type(supplied[name]) is not type(value) or supplied[name] != value):
            raise ValueError("Conflicting RTX Spark setting: " + name)
        supplied[name] = value
    definitions = {MODEL.name: MODEL}
    for option in declarations:
        if not isinstance(option, Option) or option.name in definitions:
            raise ValueError("Invalid or duplicate option declaration")
        definitions[option.name] = option
    unknown = set(supplied) - definitions.keys()
    if unknown:
        raise ValueError("Unknown RTX Spark setting: " + ", ".join(sorted(unknown)))
    resolved = {}
    for name, option in definitions.items():
        value = supplied[name] if name in supplied else option.default
        if value is MISSING:
            continue
        if name not in supplied:
            value = deepcopy(value)
        if type(value) is not option.value_type:
            raise TypeError("rtx_spark." + name + " must be " + option.value_type.__name__)
        resolved[name] = value
    return resolved


def validate_options(arguments, options, declarations):
    for option in declarations:
        if option.name in options and option.validate is not None:
            option.validate(arguments, options[option.name])


def apply_options(arguments, options, declarations):
    """Apply validated declarations in their recipe-defined order."""
    for option in declarations:
        if option.name in options and option.apply is not None:
            option.apply(arguments, options[option.name])
