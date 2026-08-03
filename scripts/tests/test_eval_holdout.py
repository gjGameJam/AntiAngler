"""Offline tests for scripts/eval_holdout.py + scripts/conf_sweep.py (ROADMAP W5).

These two scripts are the PROMOTION GATE: their center-distance matching, size strata and
bootstrap CI decide whether a retrained detector replaces `best.pt`/`best_sar.pt`. Until W5 they
had no tests at all, because both imported ultralytics at module top and CI pip-installs nothing.
W5 moved that import inside the inference helpers (`load_yolo()`), so everything that is NOT
inference — tree selection, weight resolution, subset building, matching, bootstrapping — is now
assertable here. Pure stdlib, no network, no ML deps, no dependence on the repo's data/ tree
(the filesystem-facing tests build their own throwaway trees).

Run either of:
    python scripts/tests/test_eval_holdout.py
    python -m unittest scripts.tests.test_eval_holdout
"""
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import conf_sweep  # noqa: E402
import eval_holdout as E  # noqa: E402


def make_tree(root, split_chips, base_photos=(), weights=()):
    """Build a throwaway YOLO tree: {split: [(chip_stem, [label_lines])]} -> images/ + labels/."""
    root = Path(root)
    for split, chips in split_chips.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem, lines in chips:
            (root / "images" / split / f"{stem}.png").write_bytes(b"")
            if lines is not None:
                (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
    for split, stems in base_photos:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            (root / "images" / split / f"{stem}.png").write_bytes(b"")
    if weights:
        (root / "weights").mkdir(parents=True, exist_ok=True)
        for name in weights:
            (root / "weights" / name).write_bytes(b"")
    return root


class TestImportStaysStdlibOnly(unittest.TestCase):
    """CI runs the suite with NO pip install, so importing these must not pull ultralytics/torch.

    W5 moved `from ultralytics import YOLO` into load_yolo(); this is the property that would
    break silently if someone moved it back to a module top — and it would take the whole gate's
    test coverage with it (collection error, not a clear failure)."""

    STDLIB_ONLY = ("eval_holdout", "conf_sweep")

    def test_modules_import_without_ml_deps(self):
        import builtins
        import importlib

        blocked = ("ultralytics", "torch", "torchvision", "cv2")
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                raise ImportError(f"blocked for this test: {name}")
            return real_import(name, *args, **kwargs)

        saved = {m: sys.modules.pop(m, None) for m in self.STDLIB_ONLY}
        builtins.__import__ = guard
        try:
            for mod in self.STDLIB_ONLY:
                with self.subTest(module=mod):
                    importlib.import_module(mod)
        finally:
            builtins.__import__ = real_import
            for mod, prev in saved.items():
                if prev is not None:
                    sys.modules[mod] = prev
                else:
                    sys.modules.pop(mod, None)

    def test_load_yolo_is_the_only_ml_entry_point(self):
        """Every ultralytics reference must sit inside a function, not at module scope."""
        for mod in self.STDLIB_ONLY:
            src = (SCRIPTS_DIR / f"{mod}.py").read_text().splitlines()
            offenders = [l for l in src
                         if l.startswith(("import ultralytics", "from ultralytics", "import torch"))]
            self.assertEqual(offenders, [], f"{mod}.py has a module-level ML import: {offenders}")


class TestPromotedByTree(unittest.TestCase):
    """--train-dir picks the DATA; nothing on disk links a tree to its checkpoint, so the table does."""

    def test_registered_trees_resolve(self):
        self.assertEqual(E.promoted_weights(E.REPO / "data" / "training").name, "best.pt")
        self.assertEqual(E.promoted_weights(E.REPO / "data" / "training_sar").name, "best_sar.pt")

    def test_both_modalities_share_one_weights_dir(self):
        """The whole reason the table exists: best.pt and best_sar.pt live side by side."""
        for tree in ("data/training", "data/training_sar"):
            self.assertEqual(E.promoted_weights(E.REPO / tree).parent, E.WEIGHTS)

    def test_sar_tree_key_is_the_documented_one(self):
        """Guard against a silent rename: docs/STATUS.md + build_dataset recipes use this path."""
        self.assertEqual(E.PROMOTED_BY_TREE.get("data/training_sar"), "best_sar.pt")

    def test_unregistered_tree_returns_none(self):
        self.assertIsNone(E.promoted_weights(E.REPO / "data" / "training_lidar"))

    def test_tree_outside_repo_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(E.promoted_weights(Path(tmp) / "training"))


class TestResolveWeights(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        # .resolve(): on Windows the temp dir is the 8.3 short form (GRANTB~1), and
        # promoted_weights() resolves before relative_to(REPO) — mismatched forms would fail.
        self.tmp = Path(self._td.name).resolve()
        self._saved = (E.REPO, E.WEIGHTS, dict(E.PROMOTED_BY_TREE))
        # Re-point the module at a throwaway repo so these assertions never depend on data/ .
        E.REPO = self.tmp
        E.WEIGHTS = self.tmp / "data" / "training" / "weights"
        E.WEIGHTS.mkdir(parents=True)
        (E.WEIGHTS / "best.pt").write_bytes(b"")
        (E.WEIGHTS / "best_sar.pt").write_bytes(b"")
        (self.tmp / "data" / "training_sar").mkdir(parents=True)

    def tearDown(self):
        E.REPO, E.WEIGHTS, E.PROMOTED_BY_TREE = self._saved
        self._td.cleanup()

    def test_explicit_path_wins(self):
        w = self.tmp / "data" / "training" / "weights" / "best_sar.pt"
        got = E.resolve_weights(w, self.tmp / "data" / "training", "--new")
        self.assertEqual(got, w)

    def test_explicit_missing_path_exits(self):
        """A typo'd --new used to be dropped silently -> the run compared OLD against itself."""
        with self.assertRaises(SystemExit) as cm:
            E.resolve_weights(self.tmp / "nope.pt", self.tmp / "data" / "training", "--new")
        self.assertIn("--new", str(cm.exception))
        self.assertIn("nope.pt", str(cm.exception))

    def test_default_follows_the_tree(self):
        """The core W5 property: the SAR tree must NOT default to the optical checkpoint."""
        opt = E.resolve_weights(None, self.tmp / "data" / "training", "--old")
        sar = E.resolve_weights(None, self.tmp / "data" / "training_sar", "--old")
        self.assertEqual(opt.name, "best.pt")
        self.assertEqual(sar.name, "best_sar.pt")
        self.assertNotEqual(opt, sar)

    def test_unregistered_tree_exits_with_inventory(self):
        (self.tmp / "data" / "training_lidar").mkdir(parents=True)
        with self.assertRaises(SystemExit) as cm:
            E.resolve_weights(None, self.tmp / "data" / "training_lidar", "--old")
        msg = str(cm.exception)
        self.assertIn("PROMOTED_BY_TREE", msg)
        self.assertIn("best_sar.pt", msg)  # the inventory helps the operator pick

    def test_registered_but_absent_checkpoint_exits(self):
        (E.WEIGHTS / "best_sar.pt").unlink()
        with self.assertRaises(SystemExit) as cm:
            E.resolve_weights(None, self.tmp / "data" / "training_sar", "--old")
        self.assertIn("missing", str(cm.exception))


class TestBuildSubset(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        # .resolve(): on Windows the temp dir is the 8.3 short form (GRANTB~1), and
        # promoted_weights() resolves before relative_to(REPO) — mismatched forms would fail.
        self.tmp = Path(self._td.name).resolve()
        self.tree = make_tree(
            self.tmp / "tree",
            {"test": [("2026-07-17T20-57-46Z_s1deep-gibraltar__chip_r00000_c00000", ["0 0.5 0.5 0.01 0.01"]),
                      ("2026-07-17T20-57-46Z_s1deep-gibraltar__chip_r00000_c00576", ["0 0.2 0.2 0.01 0.01",
                                                                                     "0 0.7 0.7 0.02 0.02"]),
                      ("2026-07-17T20-57-46Z_s1deep-gibraltar__chip_r00576_c00000", []),  # hard negative
                      ("2026-07-26T16-31-11Z_s1deep-singapore__chip_r00000_c00000", ["0 0.5 0.5 0.01 0.01"])],
             "train": [("2026-07-17T20-56-26Z_s1deep-fujairah__chip_r00000_c00000", [])]},
            base_photos=[("test", ["boat0001", "boat0002"])])  # base photos: no '__', must be excluded

    def tearDown(self):
        self._td.cleanup()

    def test_filters_to_the_named_scene(self):
        _, imgs, boxes = E.build_subset(self.tree, self.tmp / "sub", "test", "s1deep-gibraltar")
        self.assertEqual(len(imgs), 3)
        self.assertEqual(boxes, 3)
        self.assertTrue(all("gibraltar" in p.name for p in imgs))

    def test_base_photos_are_never_swept_in(self):
        """The '*__*.png' glob is what keeps the ~600 base boat photos out of a scene subset."""
        _, imgs, _ = E.build_subset(self.tree, self.tmp / "sub", "test", "")
        self.assertNotIn("boat0001.png", [p.name for p in imgs])
        self.assertEqual(len(imgs), 4)

    def test_copies_images_and_labels(self):
        yml, imgs, _ = E.build_subset(self.tree, self.tmp / "sub", "test", "s1deep-gibraltar")
        dst = self.tmp / "sub"
        self.assertEqual(len(list((dst / "images" / "val").glob("*.png"))), 3)
        self.assertEqual(len(list((dst / "labels" / "val").glob("*.txt"))), 3)
        self.assertEqual(yml, dst / "data.yml")

    def test_data_yml_points_at_the_subset(self):
        yml, _, _ = E.build_subset(self.tree, self.tmp / "sub", "test", "s1deep-gibraltar")
        text = yml.read_text()
        self.assertIn(f"path: {(self.tmp / 'sub').resolve().as_posix()}", text)
        self.assertIn("val: images/val", text)
        self.assertIn("nc: 1", text)

    def test_empty_label_file_counts_zero_boxes(self):
        """A reviewed empty chip is a hard negative — it must count as a chip but not a box."""
        _, imgs, boxes = E.build_subset(self.tree, self.tmp / "sub", "test", "chip_r00576_c00000")
        self.assertEqual((len(imgs), boxes), (1, 0))

    def test_no_match_returns_empty(self):
        _, imgs, boxes = E.build_subset(self.tree, self.tmp / "sub", "test", "kornati")
        self.assertEqual((imgs, boxes), ([], 0))

    def test_missing_split_dir_is_not_an_error(self):
        _, imgs, _ = E.build_subset(self.tree, self.tmp / "sub", "val", "anything")
        self.assertEqual(imgs, [])

    def test_result_is_sorted(self):
        _, imgs, _ = E.build_subset(self.tree, self.tmp / "sub", "test", "s1deep")
        self.assertEqual([p.name for p in imgs], sorted(p.name for p in imgs))


class TestSceneNames(unittest.TestCase):
    """Powers the 'you asked for a scene that isn't here' error — the zero-chip guard's payload."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        # .resolve(): on Windows the temp dir is the 8.3 short form (GRANTB~1), and
        # promoted_weights() resolves before relative_to(REPO) — mismatched forms would fail.
        self.tmp = Path(self._td.name).resolve()
        self.tree = make_tree(self.tmp / "tree", {
            "test": [("2026-07-17T20-57-46Z_s1deep-gibraltar__chip_a", []),
                     ("2026-07-17T20-57-46Z_s1deep-gibraltar__chip_b", [])],
            "train": [("2026-07-17T20-56-26Z_s1deep-fujairah__chip_a", []),
                      ("2026-07-26T16-31-11Z_s1deep-singapore__chip_a", [])]},
            base_photos=[("test", ["boat0001"])])

    def tearDown(self):
        self._td.cleanup()

    def test_distinct_sorted_prefixes(self):
        self.assertEqual(E.scene_names(self.tree, "test"),
                         ["2026-07-17T20-57-46Z_s1deep-gibraltar"])
        self.assertEqual(E.scene_names(self.tree, "train"),
                         ["2026-07-17T20-56-26Z_s1deep-fujairah", "2026-07-26T16-31-11Z_s1deep-singapore"])

    def test_missing_dir_is_empty(self):
        self.assertEqual(E.scene_names(self.tree, "val"), [])


class TestGtBoxes(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        # .resolve(): on Windows the temp dir is the 8.3 short form (GRANTB~1), and
        # promoted_weights() resolves before relative_to(REPO) — mismatched forms would fail.
        self.tmp = Path(self._td.name).resolve()

    def tearDown(self):
        self._td.cleanup()

    def _write(self, text):
        p = self.tmp / "a.txt"
        p.write_text(text)
        return p

    def test_denormalizes_to_tile_pixels(self):
        (cx, cy, size), = E.gt_boxes(self._write("0 0.5 0.5 0.01 0.02\n"))
        self.assertAlmostEqual(cx, 320.0)
        self.assertAlmostEqual(cy, 320.0)
        self.assertAlmostEqual(size, 0.02 * E.TILE)  # size = max(w, h), the strata key

    def test_missing_file_is_empty(self):
        self.assertEqual(E.gt_boxes(self.tmp / "nope.txt"), [])

    def test_blank_and_short_lines_skipped(self):
        got = E.gt_boxes(self._write("0 0.5 0.5 0.01 0.01\n\n0 0.5 0.5\nbad\n"))
        self.assertEqual(len(got), 1)

    def test_empty_file_is_a_hard_negative(self):
        self.assertEqual(E.gt_boxes(self._write("")), [])


class TestLabelFor(unittest.TestCase):
    def test_swaps_images_for_labels(self):
        self.assertEqual(E.label_for(Path("d/images/test/x.png")), Path("d/labels/test/x.txt"))

    def test_split_name_is_irrelevant(self):
        self.assertEqual(E.label_for(Path("d/images/val/x.png")), Path("d/labels/val/x.txt"))

    def test_swaps_the_last_images_segment(self):
        """A tree nested under a dir literally called 'images' must not corrupt the mapping."""
        self.assertEqual(E.label_for(Path("images/t/images/test/x.png")),
                         Path("images/t/labels/test/x.txt"))

    def test_no_images_segment_just_changes_suffix(self):
        self.assertEqual(E.label_for(Path("d/test/x.png")), Path("d/test/x.txt"))


class TestMatch(unittest.TestCase):
    """Center-distance matching replaces IoU because a 1 px shift flips a 1-5 px vessel's IoU."""

    def test_coincident_matches(self):
        used_g, used_p = E.match([(10.0, 10.0, 2.0)], [(10.0, 10.0)], 8.0)
        self.assertEqual((used_g, used_p), ({0}, {0}))

    def test_beyond_tolerance_does_not_match(self):
        used_g, used_p = E.match([(0.0, 0.0, 2.0)], [(20.0, 0.0)], 8.0)
        self.assertEqual((used_g, used_p), (set(), set()))

    def test_tolerance_is_inclusive(self):
        used_g, _ = E.match([(0.0, 0.0, 2.0)], [(8.0, 0.0)], 8.0)
        self.assertEqual(used_g, {0})

    def test_one_prediction_cannot_match_two_gts(self):
        """Otherwise a single blob over a cluster would score two TPs and inflate recall."""
        used_g, used_p = E.match([(0.0, 0.0, 2.0), (2.0, 0.0, 2.0)], [(1.0, 0.0)], 8.0)
        self.assertEqual((len(used_g), len(used_p)), (1, 1))

    def test_nearest_pairing_wins(self):
        gts = [(0.0, 0.0, 2.0), (10.0, 0.0, 2.0)]
        preds = [(9.5, 0.0), (0.5, 0.0)]
        used_g, used_p = E.match(gts, preds, 8.0)
        self.assertEqual((used_g, used_p), ({0, 1}, {0, 1}))

    def test_empty_inputs(self):
        self.assertEqual(E.match([], [(1.0, 1.0)], 8.0), (set(), set()))
        self.assertEqual(E.match([(1.0, 1.0, 2.0)], [], 8.0), (set(), set()))


class TestBootstrapRecall(unittest.TestCase):
    def test_empty_is_zero_interval(self):
        self.assertEqual(E.bootstrap_recall([]), (0.0, 0.0))

    def test_perfect_recall_has_degenerate_ci(self):
        lo, hi = E.bootstrap_recall([(3, 0, 0), (2, 0, 0)], n=200)
        self.assertEqual((lo, hi), (1.0, 1.0))

    def test_zero_recall_has_degenerate_ci(self):
        lo, hi = E.bootstrap_recall([(0, 1, 3), (0, 0, 2)], n=200)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_interval_is_ordered_and_bounded(self):
        per_chip = [(2, 1, 1), (0, 0, 3), (5, 2, 0), (1, 0, 4)]
        lo, hi = E.bootstrap_recall(per_chip, n=500)
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_seeded_and_reproducible(self):
        """Promotion decisions quote these CIs, so two runs on one model must not disagree."""
        per_chip = [(2, 1, 1), (0, 0, 3), (5, 2, 0), (1, 0, 4)]
        self.assertEqual(E.bootstrap_recall(per_chip, n=300), E.bootstrap_recall(per_chip, n=300))

    def test_chips_with_no_gt_do_not_shift_recall(self):
        """FP-only chips carry no GT, so they must leave the recall interval alone."""
        pos = [(2, 0, 2), (1, 0, 1)]
        self.assertEqual(E.bootstrap_recall(pos, n=300), E.bootstrap_recall(pos, n=300))


class TestParseArgs(unittest.TestCase):
    def test_eval_holdout_defaults_to_the_optical_tree(self):
        a = E.parse_args([])
        self.assertEqual(a.train_dir, E.TRAIN)
        self.assertIsNone(a.old)
        self.assertIsNone(a.new)
        self.assertEqual(a.holdout, "fallbacktest")

    def test_eval_holdout_accepts_the_sar_tree(self):
        a = E.parse_args(["--train-dir", "data/training_sar", "--holdout", "s1deep-gibraltar",
                          "--imgsz", "1024"])
        self.assertEqual(a.train_dir, Path("data/training_sar"))
        self.assertEqual(a.holdout, "s1deep-gibraltar")
        self.assertEqual(a.imgsz, 1024)

    def test_eval_holdout_fp_scenes_can_be_emptied(self):
        self.assertEqual(E.parse_args(["--fp-scenes"]).fp_scenes, [])

    def test_conf_sweep_defaults_to_the_optical_tree(self):
        a = conf_sweep.parse_args([])
        self.assertEqual(a.train_dir, E.TRAIN)
        self.assertIsNone(a.weights)

    def test_conf_sweep_accepts_the_sar_tree(self):
        a = conf_sweep.parse_args(["--train-dir", "data/training_sar", "--holdout", "s1deep-gibraltar",
                                   "--fp-scene", "s1deep-fujairah"])
        self.assertEqual(a.train_dir, Path("data/training_sar"))
        self.assertEqual(a.fp_scene, "s1deep-fujairah")

    def test_both_scripts_share_one_tree_default(self):
        self.assertEqual(E.parse_args([]).train_dir, conf_sweep.parse_args([]).train_dir)


class TestMainGuards(unittest.TestCase):
    """main() must fail loudly before touching the GPU when the tree/holdout is wrong."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        # .resolve(): on Windows the temp dir is the 8.3 short form (GRANTB~1), and
        # promoted_weights() resolves before relative_to(REPO) — mismatched forms would fail.
        self.tmp = Path(self._td.name).resolve()

    def tearDown(self):
        self._td.cleanup()

    def test_missing_train_dir_exits(self):
        for mod in (E, conf_sweep):
            with self.subTest(module=mod.__name__), self.assertRaises(SystemExit) as cm:
                mod.main(["--train-dir", str(self.tmp / "nothing")])
            self.assertIn("images/", str(cm.exception).replace("\\", "/"))

    def test_zero_chip_holdout_exits_listing_scenes(self):
        """The silent-empty-gate footgun: an unmatched --holdout must never evaluate nothing."""
        tree = make_tree(self.tmp / "tree", {
            "test": [("2026-07-17T20-57-46Z_s1deep-gibraltar__chip_a", [])],
            "train": [("2026-07-17T20-56-26Z_s1deep-fujairah__chip_a", [])]})
        for mod in (E, conf_sweep):
            with self.subTest(module=mod.__name__), self.assertRaises(SystemExit) as cm:
                mod.main(["--train-dir", str(tree), "--holdout", "fallbacktest"])
            msg = str(cm.exception)
            self.assertIn("matched no chips", msg)
            self.assertIn("s1deep-gibraltar", msg)  # tells the operator what IS there


if __name__ == "__main__":
    unittest.main(verbosity=2)
