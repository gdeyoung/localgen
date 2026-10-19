"""Enhance module logic tests — no binaries, no GPU, no ffmpeg."""
import unittest

from localgen.enhance_local import (_fps_of, _renumber, _rife_cmd,
                                     _rsg_cmd)


class TestEnhanceLogic(unittest.TestCase):
    def test_rsg_cmd_flags(self):
        cmd = _rsg_cmd("/t/realesrgan-ncnn-vulkan", "in", "out",
                       "realesr-animevideov3", 2)
        self.assertEqual(cmd[0], "/t/realesrgan-ncnn-vulkan")
        for flag, val in (("-i", "in"), ("-o", "out"),
                          ("-n", "realesr-animevideov3"), ("-s", "2")):
            i = cmd.index(flag)
            self.assertEqual(cmd[i + 1], val)

    def test_rife_cmd_target_and_model_dir(self):
        cmd = _rife_cmd("/t/dir/rife-ncnn-vulkan", "in", "out", "rife-v4.6", 114)
        self.assertIn("-n", cmd)
        self.assertEqual(cmd[cmd.index("-n") + 1], "114")
        self.assertEqual(cmd[cmd.index("-m") + 1], "/t/dir/rife-v4.6")

    def test_fps_fraction_parsing(self):
        self.assertEqual(_fps_of("25/1"), 25.0)
        self.assertAlmostEqual(_fps_of("30000/1001"), 29.97, places=2)
        self.assertEqual(_fps_of("30"), 30.0)

    def test_renumber_normalizes_tool_output(self):
        import tempfile, os, shutil
        d = tempfile.mkdtemp()
        try:
            # ncnn tools emit zero-padded names of varying width
            for name in ("1.png", "2.png", "10.png"):
                open(os.path.join(d, name), "wb").close()
            _renumber(d)
            self.assertEqual(sorted(os.listdir(d)),
                             ["000001.png", "000002.png", "000003.png"])
        finally:
            shutil.rmtree(d)

    def test_renumber_is_idempotent(self):
        import tempfile, os, shutil
        d = tempfile.mkdtemp()
        try:
            for i in range(1, 4):
                open(os.path.join(d, f"{i:06d}.png"), "wb").close()
            _renumber(d)
            self.assertEqual(sorted(os.listdir(d)),
                             ["000001.png", "000002.png", "000003.png"])
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
