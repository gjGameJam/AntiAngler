"""Offline tests for per-chip despeckling at INFERENCE (ROADMAP W12).

The failure guarded against here is silent and expensive: a model trained on despeckled chips that is
fed RAW chips is out of domain, and nothing crashes -- it just detects less. Measured in W12: feeding
the raw-trained `sar-deep4` despeckled chips drops Gibraltar recall 0.609 -> 0.217. So the load-bearing
property is EQUIVALENCE -- the inference path must reproduce, byte for byte, what built the training
tree -- plus "off by default", since every SAR run written before 2026-08-06 holds raw chips.

Two tiers, because `.github/workflows/tests.yml` installs NOTHING:
  * stdlib-only structural guards (always run, including in CI) -- assert the wiring is present:
    s1.py delegates to sar_despeckle, sat_fetch calls the hook inside the tiling loop and records the
    setting in the manifest. These catch the refactor that quietly drops the call.
  * numeric equivalence tests, skipped unless numpy/PIL are importable (they are locally, not in CI).
"""
import re
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

try:  # heavy-ish deps: present locally, absent in the no-pip-install CI job
    import numpy as np
    from sar_despeckle import despeckle_array, despeckle_chip_chw
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - exercised only in the dependency-free CI
    HAVE_NUMPY = False

PEAK = {"despeckle": "peak", "despeckle_window": 7, "despeckle_peak_sigma": 2.0}


def chw(seed=0, c=3, h=48, w=48):
    return np.random.default_rng(seed).integers(0, 256, size=(c, h, w), dtype=np.uint8)


class TestWiringGuards(unittest.TestCase):
    """Pure stdlib source inspection -- runs in CI, where the providers cannot even be imported."""

    def test_s1_provider_delegates_to_sar_despeckle(self):
        src = (SCRIPTS / "providers" / "s1.py").read_text(encoding="utf-8")
        self.assertIn("def postprocess_chip", src, "s1.py lost its postprocess_chip override")
        self.assertIn("despeckle_chip_chw", src,
                      "s1.py must delegate to sar_despeckle.despeckle_chip_chw so train and "
                      "inference run the SAME code")

    def test_base_provider_hook_is_a_no_op_not_abstract(self):
        src = (SCRIPTS / "providers" / "base.py").read_text(encoding="utf-8")
        self.assertIn("def postprocess_chip", src)
        hook = src[src.index("def postprocess_chip"):]
        self.assertIn("return chip", hook, "base hook must default to identity")
        head = src[:src.index("def postprocess_chip")]
        self.assertFalse(head.rstrip().endswith("@abc.abstractmethod"),
                         "postprocess_chip must NOT be abstract; existing providers rely on the no-op")

    def test_sat_fetch_calls_the_hook_and_records_the_setting(self):
        src = (SCRIPTS / "sat_fetch.py").read_text(encoding="utf-8")
        self.assertIn("provider.postprocess_chip(", src, "sat_fetch stopped calling the chip hook")
        self.assertIn('"chip_postprocess"', src,
                      "the manifest must record which preprocessing the chips got, or a run cannot "
                      "be matched to the right checkpoint")
        self.assertIn("--despeckle", src, "the --despeckle flag disappeared")
        # the call must sit inside the tiling loop, i.e. AFTER iter_chips yields a chip
        loop = src.index("for r, c, chip, sub in iter_chips")
        self.assertGreater(src.index("provider.postprocess_chip(", loop), loop,
                           "postprocess_chip must be applied per chip, after tiling")

    def test_despeckle_defaults_to_off_in_the_cli(self):
        src = (SCRIPTS / "sat_fetch.py").read_text(encoding="utf-8")
        m = re.search(r'"--despeckle".*?default=env_or\("SAT_DESPECKLE", "([a-z]+)"\)', src, re.S)
        self.assertIsNotNone(m, "could not find the --despeckle default")
        self.assertEqual(m.group(1), "off",
                         "must default OFF: the SAR runs on disk and any raw-trained checkpoint "
                         "would silently go out of domain")


@unittest.skipUnless(HAVE_NUMPY, "numpy/PIL not installed (the CI job runs stdlib-only)")
class TestTrainInferenceEquivalence(unittest.TestCase):
    def test_matches_the_training_transform_byte_for_byte(self):
        for seed in range(4):
            chip = chw(seed)
            expect = despeckle_array(np.transpose(chip, (1, 2, 0)), 7, "peak", 2.0)
            got = np.transpose(despeckle_chip_chw(chip, PEAK), (1, 2, 0))
            self.assertTrue(np.array_equal(expect, got),
                            f"seed {seed}: inference preprocessing diverged from training")

    def test_shape_dtype_and_contiguity_preserved(self):
        chip = chw()
        out = despeckle_chip_chw(chip, PEAK)
        self.assertEqual(out.shape, chip.shape)
        self.assertEqual(out.dtype, np.uint8)
        # rasterio writes this array straight out; a non-contiguous view is a silent trap
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_lee_mode_is_reachable_and_differs_from_peak(self):
        chip = chw(7)
        peak = despeckle_chip_chw(chip, PEAK)
        lee = despeckle_chip_chw(chip, {**PEAK, "despeckle": "lee"})
        self.assertFalse(np.array_equal(peak, lee))
        expect = despeckle_array(np.transpose(chip, (1, 2, 0)), 7, "lee", 2.0)
        self.assertTrue(np.array_equal(expect, np.transpose(lee, (1, 2, 0))))

    def test_window_and_sigma_are_actually_plumbed(self):
        chip = chw(3)
        base = despeckle_chip_chw(chip, PEAK)
        self.assertFalse(np.array_equal(base, despeckle_chip_chw(chip, {**PEAK, "despeckle_window": 3})))
        self.assertFalse(np.array_equal(base, despeckle_chip_chw(chip, {**PEAK, "despeckle_peak_sigma": 0.5})))


@unittest.skipUnless(HAVE_NUMPY, "numpy/PIL not installed (the CI job runs stdlib-only)")
class TestDefaultsOff(unittest.TestCase):
    def test_off_and_absent_keys_are_true_no_ops(self):
        chip = chw()
        for opts in ({"despeckle": "off"}, {}, None, {"despeckle": None}, {"despeckle_window": 7}):
            self.assertIs(despeckle_chip_chw(chip, opts), chip, f"opts={opts!r} should be a no-op")


@unittest.skipUnless(HAVE_NUMPY, "numpy/PIL not installed (the CI job runs stdlib-only)")
class TestNodataAndTargets(unittest.TestCase):
    def test_all_zero_chip_stays_all_zero(self):
        chip = np.zeros((3, 32, 32), dtype=np.uint8)
        self.assertTrue(np.array_equal(despeckle_chip_chw(chip, PEAK), chip))

    def test_saturated_target_survives_peak_mode(self):
        """The whole point of 'peak': a compact bright vessel must not be smoothed away."""
        chip = np.full((3, 40, 40), 120, dtype=np.uint8)
        chip[:, 20:22, 20:22] = 255
        out = despeckle_chip_chw(chip, PEAK)
        self.assertEqual(int(out[:, 20:22, 20:22].min()), 255,
                         "peak-preserving mode smoothed a saturated target")


if __name__ == "__main__":
    unittest.main(verbosity=2)
