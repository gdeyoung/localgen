import unittest

from localgen.tts_local import MASTER_FILTER, TRIM_FILTER, chunk_script, seeds_for
from localgen.registry import plan


class Chunking(unittest.TestCase):
    def test_sentence_boundaries(self):
        text = "One sentence here. Another follows! A third? Done."
        chunks = chunk_script(text)
        self.assertEqual("".join(c.replace(" ", "") for c in chunks), text.replace(" ", ""))

    def test_max_window_respected(self):
        text = ". ".join(["word " * 10 for _ in range(10)]).strip() + "."
        for c in chunk_script(text):
            self.assertLessEqual(len(c.split()) * 0.42, 9.0 + 0.42)

    def test_single_sentence_short(self):
        self.assertEqual(chunk_script("Hi there."), ["Hi there."])


class Seeds(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(seeds_for(3), seeds_for(3))

    def test_offset_separates_scenes(self):
        base = seeds_for(2)
        off = seeds_for(2, offset=1)
        self.assertNotEqual(base, off)
        self.assertEqual(off[0] - base[0], 10)


class Chains(unittest.TestCase):
    def test_master_chain_shape(self):
        self.assertIn("loudnorm=I=-18:TP=-2.0:LRA=9", MASTER_FILTER)
        self.assertIn("deesser", MASTER_FILTER)

    def test_trim_is_bilateral(self):
        self.assertEqual(TRIM_FILTER.count("silenceremove"), 2)
        self.assertEqual(TRIM_FILTER.count("areverse"), 2)


class Registry(unittest.TestCase):
    def test_plan_names_gaps_not_exceptions(self):
        p = plan()
        for cap in ("image", "tts", "transcribe"):
            self.assertIn(cap, p)
            self.assertIn("ok", p[cap])
            self.assertIn("missing", p[cap])


if __name__ == "__main__":
    unittest.main()
