"""Imagery-provider registry for sat_fetch.py.

``--provider {pc,s1,...}`` selects a ``Provider`` (see ``base.py``). ``pc`` is the live
Sentinel-2 / Planetary Computer optical source; ``s1`` is Sentinel-1 SAR (also free on
Planetary Computer). Both are free — no credentials. Add a source by subclassing ``Provider``
in a new module here and calling ``register(YourProvider)`` below. See ``docs/ROADMAP.md``.

This package has no dependency on ``sat_fetch`` (the orchestrator imports it, not the other
way round), so the registry stays a clean leaf.
"""
from .base import Provider
from .pc import PlanetaryComputerS2Provider
from .s1 import Sentinel1GrdProvider

_REGISTRY = {}


def register(provider_cls):
    """Add a Provider subclass to the registry, keyed by its ``.key``."""
    key = provider_cls.key
    if not key:
        raise ValueError(f"{provider_cls.__name__} must set a non-empty .key")
    _REGISTRY[key] = provider_cls


def available_providers():
    """Registered ``--provider`` keys, sorted (used for the CLI choices + error messages)."""
    return sorted(_REGISTRY)


def get_provider(key):
    """Instantiate the provider registered under ``key``. Raises RuntimeError if unknown."""
    try:
        provider_cls = _REGISTRY[key]
    except KeyError:
        raise RuntimeError(
            f"Unknown --provider {key!r}. Available: {', '.join(available_providers())}.")
    return provider_cls()


register(PlanetaryComputerS2Provider)
register(Sentinel1GrdProvider)
