"""The Provider interface every imagery source implements.

A provider owns the four *source-specific* steps — catalog auth, scene search, the
AOI cloud/coverage probe, and the reflectance/backscatter -> 8-bit conversion — plus a
windowed AOI reader. Everything else (AOI resolution, the scene-selection loop, tiling,
chip/manifest IO) lives in ``sat_fetch.py`` (the orchestrator) and is identical across
providers, so the run-dir / manifest data contract the downstream stages read never changes.

To add a provider (see ``docs/ROADMAP.md`` Phases 1-2):
  1. subclass ``Provider``, set the metadata class attributes, implement the five methods;
  2. register it in ``providers/__init__.py``.

Reuse ``providers._raster`` (``decimated_read`` / ``gdal_env``) for the windowed COG read —
those are provider-agnostic. See ``providers/pc.py`` (optical) and ``providers/s1.py`` (SAR)
for the reference implementations, which show exactly which fields each method swaps.
"""
import abc


class Provider(abc.ABC):
    """One imagery source behind the sat_fetch seam.

    The class attributes below are metadata the orchestrator reads: ``main()`` fills any
    unset CLI knob (``--collection`` / ``--bands`` / ``--resolution`` / ``--stretch-max``)
    from the ``default_*`` values, and stamps ``manifest_tag`` / ``known_limitations`` into
    the run manifest. Keep ``__init__`` cheap and side-effect-free — a provider is
    instantiated just to be selected; all network/IO happens inside the methods.
    """

    #: ``--provider`` value / registry key (e.g. "pc", "planet", "s1")
    key = None
    #: human-readable label for log lines
    display_name = None
    #: written verbatim into ``manifest["provider"]``
    manifest_tag = None

    #: provider CLI defaults — applied by ``main()`` only when the matching flag is unset,
    #: so ``--provider planet`` gets 3 m / BGRN, a SAR provider gets VV+VH, etc.
    default_collection = None
    default_bands = None
    default_resolution = 10.0
    default_stretch_max = 3000.0

    #: run-root subdir under ``data/raw/`` — one per MODALITY, so optical and SAR runs never
    #: share a directory (a SAR chip must never reach the optical training set — the two have
    #: separate models, training trees and gates). ``--output-dir``/``SAT_OUTPUT_DIR``
    #: still override it. Providers reading the same sensor share one value.
    output_subdir = "sentinel2"

    #: written verbatim into ``manifest["known_limitations"]``
    known_limitations = ()

    @abc.abstractmethod
    def open_catalog(self, stac_url, sub_key):
        """Open + authenticate the catalog/session and return the handle the other
        methods take as ``catalog`` (a STAC ``Client`` for the Planetary Computer
        Sentinel-2 / Sentinel-1 sources, or a source-specific session for others)."""

    @abc.abstractmethod
    def search_candidates(self, catalog, collection, bbox, start, end, max_cloud, max_search):
        """Return up to ``max_search`` candidate scene items covering ``bbox`` within
        ``[start, end]`` (WGS84 lon/lat bbox tuple; ISO ``YYYY-MM-DD`` dates), best-first.
        ``max_cloud`` is the scene-wide cloud prefilter percent (ignore it for sensors with
        no cloud concept, e.g. SAR). Raise ``RuntimeError`` if nothing matches."""

    @abc.abstractmethod
    def assess_aoi(self, item, bbox, insecure_tls, probe_res=60.0):
        """Probe one candidate over the AOI (not the whole scene). Return a dict with at
        least ``{"id", "datetime", "aoi_coverage", "aoi_cloud"}`` where ``aoi_coverage`` is
        the fraction of the AOI the scene actually contains (0-1) and ``aoi_cloud`` is the
        fraction obscured by cloud/shadow (0-1), or ``None`` if the source has no cloud
        mask. ``select_scene`` ranks candidates on these two numbers."""

    @abc.abstractmethod
    def read_aoi(self, item, bbox, bands, resolution, insecure_tls=False):
        """Windowed-read the chosen scene clipped to ``bbox`` in the scene's native CRS.
        Return ``(stack[C, H, W], transform, crs, valid_mask[H, W])`` — ``transform``/``crs``
        make each chip self-describing so downstream code maps pixels -> lat/lon directly."""

    @abc.abstractmethod
    def scale_to_uint8(self, stack, valid, mode, stretch_max, stretch_pct):
        """Convert the reflectance/backscatter ``stack`` to an 8-bit stack, stretched once
        over the whole AOI. Return ``(uint8_stack, params)`` where ``params`` records how
        (goes into the manifest so the stretch is reproducible). Match this to the detection
        model's training preprocessing."""

    # --- optional hook (NOT abstract: providers that need no per-chip step inherit the no-op) ---

    def postprocess_chip(self, chip, opts):
        """Per-chip preprocessing applied **after tiling**, on the uint8 ``(C, H, W)`` chip.

        Returns a ``(C, H, W)`` uint8 array (the same one by default). ``opts`` is the
        ``chip_postprocess`` dict ``sat_fetch.main()`` builds from the CLI; a provider reads
        only the keys it understands and ignores the rest.

        **Why after tiling, not on the AOI mosaic:** a filter with a spatial window gives
        different answers at a tile boundary than in a tile interior. The SAR training chips
        were despeckled per-chip, after tiling, with nodata already zeroed — so filtering the
        mosaic first would disagree with training within ~window/2 px of every tile edge. That
        mismatch is exactly what silently breaks a model trained on filtered data. Whatever a
        provider does here, it must match what its model was TRAINED on, byte for byte.
        """
        return chip
