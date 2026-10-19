"""localswap CLI: doctor | plan | image | video | tts"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="localswap", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="probe accelerators + backends (JSON)")
    sub.add_parser("plan", help="what this device can generate now, and what's missing")

    p_img = sub.add_parser("image", help="generate one image locally")
    p_img.add_argument("prompt")
    p_img.add_argument("out", nargs="?", default="out.png")
    p_img.add_argument("--width", type=int, default=768)
    p_img.add_argument("--height", type=int, default=432)
    p_img.add_argument("--steps", type=int, default=4)
    p_img.add_argument("--seed", type=int, default=42)
    p_img.add_argument("--device", default="auto")

    p_vid = sub.add_parser("video", help="text-to-video locally (LTX-Video; slow on iGPU)")
    p_vid.add_argument("prompt")
    p_vid.add_argument("out", nargs="?", default="out.mp4")
    p_vid.add_argument("--width", type=int, default=768)
    p_vid.add_argument("--height", type=int, default=448)
    p_vid.add_argument("--frames", type=int, default=61)
    p_vid.add_argument("--steps", type=int, default=20)
    p_vid.add_argument("--fps", type=int, default=25)
    p_vid.add_argument("--seed", type=int, default=42)
    p_vid.add_argument("--device", default="auto")
    p_vid.add_argument("--negative", default=None)

    p_up = sub.add_parser("upscale", help="neural 2x/3x/4x upscale of a video or image")
    p_up.add_argument("in_path")
    p_up.add_argument("out_path", nargs="?", default=None)
    p_up.add_argument("--factor", type=int, default=2, choices=[2, 3, 4])
    p_up.add_argument("--model", default="realesr-animevideov3")

    p_in = sub.add_parser("interp", help="RIFE frame interpolation (raise fps)")
    p_in.add_argument("in_path")
    p_in.add_argument("out_path", nargs="?", default=None)
    p_in.add_argument("--mult", type=float, default=2)
    p_in.add_argument("--model", default="rife-v4.6")

    p_tts = sub.add_parser("tts", help="narrate text in a cloned voice (your reference)")
    p_tts.add_argument("text")
    p_tts.add_argument("out", nargs="?", default="narration.wav")
    p_tts.add_argument("--ref-wav", required=True, help="reference voice WAV (yours)")
    p_tts.add_argument("--ref-text", required=True, help="verbatim transcript of the reference")
    p_tts.add_argument("--instruction", default="Narrate warmly and clearly.")
    p_tts.add_argument("--seed-base", type=int, default=1000)
    p_tts.add_argument("--cpu", action="store_true")

    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        from . import doctor
        print(json.dumps(doctor.probe(), indent=2))
        return 0
    if a.cmd == "plan":
        from . import registry
        print(json.dumps(registry.plan(), indent=2))
        return 0
    if a.cmd == "image":
        from .image_local import generate
        print(json.dumps(generate(a.prompt, a.out, width=a.width, height=a.height,
                                   steps=a.steps, seed=a.seed, device=a.device), indent=2))
        return 0
    if a.cmd == "video":
        from .video_local import generate_video
        print(json.dumps(generate_video(a.prompt, a.out, width=a.width,
                                        height=a.height, num_frames=a.frames,
                                        steps=a.steps, seed=a.seed, fps=a.fps,
                                        device=a.device,
                                        negative_prompt=a.negative), indent=2))
        return 0
    if a.cmd == "upscale":
        from .enhance_local import upscale
        out_path = a.out_path or (a.in_path.rsplit(".", 1)[0] + f"_{a.factor}x.mp4")
        print(json.dumps(upscale(a.in_path, out_path, factor=a.factor,
                                 model=a.model), indent=2))
        return 0
    if a.cmd == "interp":
        from .enhance_local import interpolate
        out_path = a.out_path or (a.in_path.rsplit(".", 1)[0] + f"_{int(a.mult)}x.mp4")
        print(json.dumps(interpolate(a.in_path, out_path, mult=a.mult,
                                     model=a.model), indent=2))
        return 0
    if a.cmd == "tts":
        from .tts_local import narrate
        print(json.dumps(narrate(a.text, a.out, ref_wav=a.ref_wav, ref_text=a.ref_text,
                                 instruction=a.instruction, seed_base=a.seed_base,
                                 cpu=a.cpu), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
