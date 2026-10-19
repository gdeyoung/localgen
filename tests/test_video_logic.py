"""Video module logic tests — no model, no GPU, no network."""
import os
import unittest
from unittest.mock import patch

from localgen.video_local import _ffmpeg_cmd, _resolve_model


class TestVideoLogic(unittest.TestCase):
    def test_ffmpeg_cmd_shape(self):
        cmd = _ffmpeg_cmd("out.mp4", 768, 448, 25)
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("rgb24", cmd)
        self.assertIn("768x448", cmd)
        # input spec before output path
        self.assertLess(cmd.index("-i"), cmd.index("out.mp4"))
        self.assertIn("libx264", cmd)
        self.assertIn("yuv420p", cmd)
        # framerate flag pairs with fps value
        i = cmd.index("-r")
        self.assertEqual(cmd[i + 1], "25")

    def test_ffmpeg_cmd_always_overwrites(self):
        self.assertIn("-y", _ffmpeg_cmd("o.mp4", 64, 64, 24))

    def test_resolve_model_env_override(self):
        with patch.dict(os.environ, {"LOCALGEN_VIDEO_MODEL": "/custom/ltx"}):
            self.assertEqual(_resolve_model(None), "/custom/ltx")
        # explicit arg wins over env
        with patch.dict(os.environ, {"LOCALGEN_VIDEO_MODEL": "/custom/ltx"}):
            self.assertEqual(_resolve_model("/explicit"), "/explicit")

    def test_resolve_model_default_path(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("LOCALGEN_VIDEO_MODEL", "LOCALGEN_HOME")}
        with patch.dict(os.environ, env, clear=True):
            expected = os.path.join(os.path.expanduser("~/.localgen"),
                                    "models", "video", "ltx-video-int8")
            self.assertEqual(_resolve_model(None), expected)


if __name__ == "__main__":
    unittest.main()
