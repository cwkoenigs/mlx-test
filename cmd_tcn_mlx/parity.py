"""Check that the MLX TCN equals the PyTorch TCN: copy the torch weights into the MLX model and
compare outputs and loss on the same block of real (synthetic) data, in eval mode.

    .venv/bin/python -m cmd_tcn_mlx.parity --data-root <cmd-out> [--channels 128] [--batch 64]

Exit 1 if the max abs output difference exceeds --tol. On the GPU the two frameworks agree to about
2e-3 on outputs of order 1 (float32 roundoff through ten dilated convolutions); the losses agree to
1e-4. ``--device cpu`` would tighten that but needs MLX's CPU conv JIT, i.e. working Xcode
command-line tools (on this Mac the macOS 27.0 SDK's .tbd files fail to link).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tol", type=float, default=5e-3, help="max abs output difference allowed (GPU float32 conv roundoff ~2e-3 at 512 ch)")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="MLX device for the check")
    args = ap.parse_args(argv)

    import mlx.core as mx
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    import torch

    from cmd_tcn.model import CMDTCNRegressor as TorchTCN
    from cmd_tcn_mlx.data import BlockReader, Normalizer, split_dir
    from cmd_tcn_mlx.model import CMDTCNRegressor as MlxTCN
    from cmd_tcn_mlx.model import masked_huber

    torch.manual_seed(0)
    tm = TorchTCN(channels=args.channels).eval()
    mm = MlxTCN(channels=args.channels)
    mm.eval()

    # ---- copy weights torch -> mlx -----------------------------------------------------------
    def t2m(t):
        return mx.array(t.detach().cpu().numpy().astype(np.float32))

    for tb, mb in zip(tm.tcn.blocks, mm.tcn.blocks):
        for tconv, mconv in ((tb.net[0].conv, mb.conv1), (tb.net[3].conv, mb.conv2)):
            w = tconv.weight.detach()                      # effective weight-normed weight [out, in, k]
            mconv.v = t2m(w.permute(0, 2, 1))              # MLX layout [out, k, in]
            mconv.g = t2m(w.flatten(1).norm(dim=1))        # g = ||w|| so g * v / ||v|| == w
            mconv.bias = t2m(tconv.bias)
        if not isinstance(tb.downsample, torch.nn.Identity):
            mb.downsample.weight = t2m(tb.downsample.weight.permute(0, 2, 1))
            mb.downsample.bias = t2m(tb.downsample.bias)
    mm.static_proj.weight, mm.static_proj.bias = t2m(tm.static[0].weight), t2m(tm.static[0].bias)
    mm.head1.weight, mm.head1.bias = t2m(tm.head[0].weight), t2m(tm.head[0].bias)
    mm.head2.weight, mm.head2.bias = t2m(tm.head[3].weight), t2m(tm.head[3].bias)
    mx.eval(mm.parameters())

    # ---- same batch ---------------------------------------------------------------------------
    tdir = split_dir(args.data_root, "train__CMD")
    norm = Normalizer.fit(tdir, 512)
    raw = BlockReader(tdir).read((0, args.batch))
    x_seq = np.clip((np.nan_to_num(raw["X_seq"]) - norm.seq_mean) / norm.seq_std, -norm.clip, norm.clip).astype(np.float32)
    x_static = np.clip((np.nan_to_num(raw["X_static"]) - norm.static_mean) / norm.static_std, -norm.clip, norm.clip).astype(np.float32)
    y = raw["y"]
    mask = (raw["outcome_enrollment_mask"] * np.isfinite(y)).astype(np.float32)
    y = np.nan_to_num(y)

    with torch.no_grad():
        tp = tm(torch.from_numpy(x_seq), torch.from_numpy(raw["enrollment_mask"]), torch.from_numpy(x_static))
        tt = torch.log1p(torch.from_numpy(y).clamp_min(0))
        tmask = torch.from_numpy(mask)
        tloss = (torch.nn.HuberLoss(reduction="none")(tp, tt) * tmask).sum() / tmask.sum().clamp_min(1)
    mp = mm(mx.array(x_seq), mx.array(raw["enrollment_mask"]), mx.array(x_static))
    mloss, _, _ = masked_huber(mp, mx.array(y), mx.array(mask))
    mx.eval(mp, mloss)

    diff = float(np.abs(tp.numpy() - np.array(mp)).max())
    scale = float(np.abs(tp.numpy()).max())
    print(f"batch {args.batch} x channels {args.channels}: max |torch - mlx| = {diff:.2e} (outputs up to {scale:.3f}); "
          f"loss torch {float(tloss):.6f} mlx {float(mloss.item()):.6f}")
    ok = diff <= args.tol and abs(float(tloss) - float(mloss.item())) <= 1e-4
    print("PARITY:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
