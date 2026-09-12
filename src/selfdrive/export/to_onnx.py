"""Export a trained policy to ONNX for the phone.

    uv run python -m selfdrive.export.to_onnx --model runs/<run>/best_model.zip

The exported graph is deliberately self-contained: observations are normalized
analytically inside the environment (see `envs/obs.py`), so there is no running-statistics
file that could drift out of sync with the Android app. One file in, one file out.

The graph also performs the final clip to [-1, 1]. SB3's deterministic action for a
Box space is the raw Gaussian mean, which is *not* bounded - the vectorized env clips it
on the way in. Baking that clip into the export means the phone cannot forget to do it
and send an out-of-range steering angle to the ESP32.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch as th


class DeterministicPolicy(th.nn.Module):
    """Observation in, clipped action out. No sampling, no value head."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.policy.extract_features(obs)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            latent_pi = self.policy.mlp_extractor.forward_actor(features[0])
        return th.clamp(self.policy.action_net(latent_pi), -1.0, 1.0)


def export(model_path: str | Path, out_path: str | Path, opset: int = 17) -> Path:
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    policy = model.policy.eval()
    obs_dim = int(model.observation_space.shape[0])

    wrapper = DeterministicPolicy(policy).eval()
    dummy = th.zeros(1, obs_dim, dtype=th.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    th.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
    )
    return out_path


def verify(model_path: str | Path, onnx_path: str | Path, n: int = 64,
           tol: float = 1e-4) -> float:
    """Compare ONNX output against SB3 on random observations.

    Worth doing every time. A silently wrong export produces a car that drives badly for
    reasons no amount of retraining will fix.
    """
    import onnxruntime as ort
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    obs_dim = int(model.observation_space.shape[0])
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1.0, 1.0, size=(n, obs_dim)).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_actions = session.run(None, {"observation": obs})[0]
    sb3_actions, _ = model.predict(obs, deterministic=True)

    max_diff = float(np.abs(onnx_actions - sb3_actions).max())
    if max_diff > tol:
        raise AssertionError(f"ONNX and SB3 disagree by {max_diff:.2e} (tolerance {tol:.0e})")
    return max_diff


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Export a trained policy to ONNX.")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=None, help="defaults to <model>.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args(argv)

    out = args.out or str(Path(args.model).with_suffix("")) + ".onnx"
    path = export(args.model, out, args.opset)
    print(f"wrote {path}")

    if not args.no_verify:
        print(f"verified against SB3, max difference {verify(args.model, path):.2e}")


if __name__ == "__main__":
    main()
