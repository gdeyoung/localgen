"""localswap — swap cloud media generation for whatever compute this device has.

A device-adaptive library: probe the hardware (Intel iGPU/GPU via OpenVINO,
Vulkan, CPU), then generate images, cloned-voice narration, and transcripts
locally. Built for AI agents (Hermes, Claude Code, any coding agent) that need
media without cloud services.

    from localswap import doctor, registry
    report = doctor.probe()
    plan = registry.plan()          # what this device can do right now
    from localswap.image_local import generate
    generate("a red bicycle on mars", "out.png", device="auto")

CLI: localswap doctor | image | tts | plan
"""

__version__ = "0.1.0"

from . import doctor, registry  # noqa: F401
