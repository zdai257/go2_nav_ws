"""Convert the Go2 digital-twin URDF to USD (Isaac Sim python) and print the resulting prim tree.

    ./run_isaac.sh scripts/convert_urdf_to_usd.py            # headless by default
    ./run_isaac.sh scripts/convert_urdf_to_usd.py --force

``isaac_nav.py`` does this automatically when the cached USD is missing or older than the URDF.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from go2nav.isaac.app import launch

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
parser.add_argument("--gui", action="store_true")
args, _ = parser.parse_known_args()
app = launch(headless=not args.gui)

from pxr import Usd, UsdPhysics

from go2nav.core.config import load_config
from go2nav.isaac.go2_asset import ensure_go2_usd

cfg = load_config()
cfg["isaac"]["robot_source"] = "urdf"
usd = ensure_go2_usd(cfg, force=args.force)
stage = Usd.Stage.Open(usd)
print(f"default prim: {stage.GetDefaultPrim().GetPath()}")
for prim in stage.Traverse():
    tags = []
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        tags.append("ArticulationRoot")
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        tags.append("RigidBody")
    if prim.IsA(UsdPhysics.RevoluteJoint):
        tags.append("RevoluteJoint")
    if tags or prim.GetPath().pathElementCount <= 2:
        print(f"  {str(prim.GetPath()):55s} {prim.GetTypeName():18s} {' '.join(tags)}")
app.close()
