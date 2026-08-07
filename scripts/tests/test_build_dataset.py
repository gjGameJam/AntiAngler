"""Offline tests for build_dataset.py -- the split assembler.

Focus: the cross-split LEAK found 2026-08-06. `purge_al_files` used to delete only files whose name
contained the `__` active-learning marker. An external ingester is free to name its chips anything --
`ingest_s2ships_finland.py` writes `finlandS2_<tile><date>_r00_c02`, with no `__` -- so those files
survived every purge while the seeded split kept reassigning them. A chip assigned to train in one
build and val in the next ended up in BOTH splits: 510 of 1034 Finland chips were duplicated when
this was caught, silently contaminating the val set used for best-epoch selection.

Pure stdlib (build_dataset imports only argparse/json/random/shutil), so this runs in CI with no
pip install -- keep it that way.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_dataset as bd  # noqa: E402

SPLITS = ("train", "val", "test")


def make_tree(root, splits_content):
    """splits_content: {split: [stem, ...]} -> create image+label files in a training tree."""
    for split in SPLITS:
        for sub in ("images", "labels"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)
    for split, stems in splits_content.items():
        for stem in stems:
            (root / "images" / split / f"{stem}.png").write_text("png")
            (root / "labels" / split / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def make_export(tag_dir, stems, empty=()):
    (tag_dir / "images").mkdir(parents=True, exist_ok=True)
    (tag_dir / "labels").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (tag_dir / "images" / f"{stem}.png").write_text("png")
        body = "" if stem in empty else "0 0.5 0.5 0.1 0.1\n"
        (tag_dir / "labels" / f"{stem}.txt").write_text(body)


class TestPurgeByProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_purges_export_chip_without_the_marker(self):
        """The regression: an export chip with no '__' in its name must still be purged."""
        data = self.tmp / "training"
        make_tree(data, {"train": ["finlandS2_34VEM_r00_c02", "boat1"],
                         "val": ["finlandS2_34VEM_r00_c04", "boat2"]})
        al_stems = {"finlandS2_34VEM_r00_c02", "finlandS2_34VEM_r00_c04"}

        removed = bd.purge_al_files(data, al_stems)

        self.assertEqual(removed, 4)  # 2 chips x (image + label)
        self.assertFalse((data / "images" / "train" / "finlandS2_34VEM_r00_c02.png").exists())
        self.assertFalse((data / "labels" / "val" / "finlandS2_34VEM_r00_c04.txt").exists())
        # base photos are never in an export tag -> untouched
        self.assertTrue((data / "images" / "train" / "boat1.png").exists())
        self.assertTrue((data / "images" / "val" / "boat2.png").exists())

    def test_still_purges_marker_named_files_not_in_the_export_set(self):
        """Legacy behaviour retained: a stale '__' file is removed even if its tag is gone."""
        data = self.tmp / "training"
        make_tree(data, {"train": ["oldrun__chip_r0_c0", "boat1"]})
        removed = bd.purge_al_files(data, al_stems=())
        self.assertEqual(removed, 2)
        self.assertFalse((data / "images" / "train" / "oldrun__chip_r0_c0.png").exists())
        self.assertTrue((data / "images" / "train" / "boat1.png").exists())

    def test_count_base_excludes_oddly_named_export_chips(self):
        """'base kept' must not count an export chip as an untouched base photo."""
        data = self.tmp / "training"
        make_tree(data, {"train": ["finlandS2_a", "boat1", "boat2"], "val": ["finlandS2_b"]})
        base = bd.count_base(data, {"finlandS2_a", "finlandS2_b"})
        self.assertEqual(base["train"], 2)
        self.assertEqual(base["val"], 0)


class TestNoCrossSplitDuplication(unittest.TestCase):
    """End-to-end: rebuilding after the stem set changes must never leave a chip in two splits."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = self.tmp / "training"
        make_tree(self.data, {"train": ["boat1"], "val": ["boat2"], "test": ["boat3"]})
        self.exports = self.tmp / "exports"

    def run_build(self, argv):
        argv = ["build_dataset.py", "--data", str(self.data), "--exports", str(self.exports)] + argv
        old = sys.argv
        sys.argv = argv
        try:
            bd.main()
        finally:
            sys.argv = old

    def placement(self):
        where = {}
        for split in SPLITS:
            for p in (self.data / "images" / split).glob("*.png"):
                where.setdefault(p.stem, set()).add(split)
        return where

    def test_rebuild_with_new_chips_does_not_duplicate_unmarked_stems(self):
        # Round 1: 20 chips named WITHOUT the '__' marker, exactly like the Finland ingester.
        first = [f"finlandS2_tileA_r{i:02d}_c00" for i in range(20)]
        make_export(self.exports / "finland-s2", first)
        self.run_build([])
        after_first = self.placement()
        self.assertTrue(all(len(v) == 1 for v in after_first.values()))

        # Round 2: add chips. The stem set changes -> the seeded shuffle reassigns many of the
        # originals to a different split. Before the fix, the stale copies stayed behind.
        make_export(self.exports / "w4", [f"run__chip_r{i:02d}_c00" for i in range(10)])
        self.run_build([])
        after_second = self.placement()

        dupes = {s: v for s, v in after_second.items() if len(v) > 1}
        self.assertEqual(dupes, {}, f"stems in more than one split: {dupes}")

        # The reshuffle must actually have moved something, or this test proves nothing.
        moved = [s for s in first if after_first.get(s) != after_second.get(s)]
        self.assertTrue(moved, "split assignment did not change; test would pass vacuously")

        # Every export chip is present exactly once, and the base photos survived.
        for stem in first:
            self.assertEqual(len(after_second.get(stem, ())), 1)
        for stem in ("boat1", "boat2", "boat3"):
            self.assertIn(stem, after_second)

    def test_images_and_labels_stay_in_lockstep(self):
        make_export(self.exports / "tag", [f"run__chip_{i}" for i in range(12)])
        self.run_build([])
        make_export(self.exports / "tag2", [f"run2__chip_{i}" for i in range(6)])
        self.run_build([])
        for split in SPLITS:
            imgs = {p.stem for p in (self.data / "images" / split).glob("*.png")}
            lbls = {p.stem for p in (self.data / "labels" / split).glob("*.txt")}
            self.assertEqual(imgs, lbls, f"{split}: image/label mismatch")

    def test_holdout_scene_chips_go_whole_into_test(self):
        make_export(self.exports / "tag",
                    [f"run__chip_{i}" for i in range(10)] + [f"kornati-adriatic__chip_{i}" for i in range(5)])
        self.run_build(["--holdout-scene", "kornati-adriatic"])
        where = self.placement()
        for i in range(5):
            self.assertEqual(where[f"kornati-adriatic__chip_{i}"], {"test"})
        for i in range(10):
            self.assertNotIn("test", where[f"run__chip_{i}"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
