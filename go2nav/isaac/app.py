"""SimulationApp bootstrap. Must run before any ``omni`` / ``isaacsim.core`` import."""

from __future__ import annotations

_APP = None

REQUIRED_EXTENSIONS = (
    "isaacsim.asset.importer.urdf",
    "isaacsim.sensors.physx",
    "isaacsim.util.debug_draw",
)


def launch(headless: bool = False, width: int = 1600, height: int = 900, extra_extensions=()):
    global _APP
    if _APP is not None:
        return _APP
    import sys

    sys.stdout.reconfigure(line_buffering=True)  # Kit's fast shutdown skips flushing piped stdout
    from isaacsim import SimulationApp

    _APP = SimulationApp({"headless": headless, "width": width, "height": height,
                          "renderer": "RaytracedLighting", "anti_aliasing": 0})
    from isaacsim.core.utils.extensions import enable_extension

    for ext in (*REQUIRED_EXTENSIONS, *extra_extensions):
        enable_extension(ext)
    _APP.update()
    return _APP
