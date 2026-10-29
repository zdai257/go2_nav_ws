"""Export an Isaac Lab pretrained Go2 velocity policy to a torch-free .npz.

The checkpoints (rsl_rl PPO) are fetched from NVIDIA's public Isaac 4.5 asset bucket:
  rough -> Isaac-Velocity-Rough-Unitree-Go2-v0 (1500 iters, 235 obs incl. height scan)  [default backend]
  flat  -> Isaac-Velocity-Flat-Unitree-Go2-v0  (300 iters, 48 obs; crouched gait, weak turning)
Only the actor MLP is kept so the walking controller runs with plain numpy inside Isaac Sim.

Needs torch (run once from any env that has it, e.g. conda base):
    python scripts/export_go2_policy.py --task rough
    python scripts/export_go2_policy.py --task flat
    python scripts/export_go2_policy.py --checkpoint my_run/model_1499.pt --out-dir assets/policies/my_go2
"""

import argparse
import os
import shutil
import urllib.request

import numpy as np

BUCKET = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/"
          "PretrainedCheckpoints/rsl_rl/")
TASKS = {"flat": "Isaac-Velocity-Flat-Unitree-Go2-v0", "rough": "Isaac-Velocity-Rough-Unitree-Go2-v0"}
POLICY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "policies")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="rough")
    parser.add_argument("--checkpoint", default=None, help="local rsl_rl checkpoint (downloads if omitted)")
    parser.add_argument("--out-dir", default=None, help="default: assets/policies/go2_<task>")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(POLICY_ROOT, f"go2_{args.task}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.pt")
    if args.checkpoint:
        if os.path.abspath(args.checkpoint) != os.path.abspath(ckpt):
            shutil.copy(args.checkpoint, ckpt)
    elif not os.path.exists(ckpt):
        url = BUCKET + TASKS[args.task] + "/checkpoint.pt"
        print(f"[export] downloading {url}")
        urllib.request.urlretrieve(url, ckpt)

    state = torch.load(ckpt, map_location="cpu")["model_state_dict"]
    layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("actor.")})
    arrays = {}
    for i, idx in enumerate(layers):
        arrays[f"W{i}"] = state[f"actor.{idx}.weight"].numpy().astype(np.float32)
        arrays[f"b{i}"] = state[f"actor.{idx}.bias"].numpy().astype(np.float32)
    out = os.path.join(out_dir, "policy.npz")
    np.savez(out, **arrays)
    print(f"[export] {len(layers)} layers -> {os.path.abspath(out)}")
    for i in range(len(layers)):
        print(f"   W{i}: {arrays[f'W{i}'].shape}")
    if not os.path.exists(os.path.join(out_dir, "policy.yaml")):
        print("[export] NOTE: write a policy.yaml next to it (copy assets/policies/go2_rough/policy.yaml)")


if __name__ == "__main__":
    main()
