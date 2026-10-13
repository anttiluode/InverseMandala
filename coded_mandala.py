"""Coded InverseMandala v2: controlled, pre-max measurements of a hidden source.

Protocol: CODED_PROTOCOL.md; frozen before scoring. All intervention codebooks
are fixed, independent of image contents; inverse sees views and rendered data,
NEVER the source nor argmax provenance. No claim of physical passive masking.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from inverse_mandala import scene, witness, psnr, warp_matrices


@dataclass(frozen=True)
class View:
    n: int = 16
    kind: str = "normal"  # normal, coded, inverted
    dx: int = 0
    dy: int = 0
    seed: int = 0

    def __post_init__(self):
        if not (1 <= self.n <= 64):
            raise ValueError("symmetry N outside 1..64")
        if self.kind not in ("normal", "coded", "inverted"):
            raise ValueError("unknown intervention")
        if self.kind == "coded" and self.seed not in (11, 29):
            raise ValueError("coded seed outside frozen codebook")


NORMAL = View()
CENTER_A = View(dx=6, dy=-4)
CENTER_B = View(dx=-5, dy=5)
INVERTED = View(kind="inverted")
CODE_A = View(kind="coded", seed=11)
CODE_B = View(kind="coded", seed=29)

ARMS: dict[str, tuple[View, ...]] = {
    "single": (NORMAL,),
    "duplicate_control": (NORMAL, NORMAL, NORMAL),
    "symmetry": (NORMAL, View(n=7), View(n=3)),
    "center": (NORMAL, CENTER_A, CENTER_B),
    "coded": (NORMAL, CODE_A, CODE_B),
    "contrast": (NORMAL, INVERTED, CENTER_A),
    "max_min": (NORMAL, INVERTED),
}


@lru_cache(maxsize=8)
def mask_bank(h: int, w: int, n: int, seed: int) -> np.ndarray:
    """Fixed 8x8 independent random transmission per branch, enlarged by NN."""
    if seed not in (11, 29):
        raise ValueError("mask seed")
    rng = np.random.default_rng(seed)
    small = rng.uniform(0.15, 1.0, size=(n, 8, 8))
    iy = np.minimum(7, np.arange(h) * 8 // h)
    ix = np.minimum(7, np.arange(w) * 8 // w)
    return np.ascontiguousarray(small[:, iy[:, None], ix[None, :]])


def layers_cv2(x: np.ndarray, view: View) -> np.ndarray:
    """Exact forward copies: OpenCV rotation, then odd-index horizontal flip."""
    a = np.asarray(x, dtype=np.float32)
    h, w = a.shape
    source = (1.0 - a) if view.kind == "inverted" else a
    center = (w // 2 + view.dx, h // 2 + view.dy)
    copies = []
    for k in range(view.n):
        rot = cv2.getRotationMatrix2D(center, k * 360.0 / view.n, 1.0)
        z = cv2.warpAffine(source, rot, (w, h))
        if k % 2:
            z = cv2.flip(z, 1)
        copies.append(z)
    stack = np.stack(copies)
    if view.kind == "coded":
        stack *= mask_bank(h, w, view.n, view.seed)
    return stack


def observe(x: np.ndarray, view: View) -> np.ndarray:
    stack = layers_cv2(x, view)
    y = stack.max(axis=0)
    return 1.0 - y if view.kind == "inverted" else y


@lru_cache(maxsize=28)
def shifted_warps(h: int, w: int, n: int, dx: int, dy: int):
    """Continuous bilinear model; differs slightly from OpenCV's 1/32 table."""
    if dx == 0 and dy == 0:
        return warp_matrices(h, w, n)
    yy, xx = np.indices((h, w), dtype=np.float64)
    indices = np.arange(h * w, dtype=np.int32)
    cx, cy = w // 2 + dx, h // 2 + dy
    out = []
    for k in range(n):
        m = cv2.getRotationMatrix2D((cx, cy), k * 360.0 / n, 1.0)
        inv = cv2.invertAffineTransform(m)
        pre_x = w - 1 - xx if k % 2 else xx
        sx = inv[0, 0] * pre_x + inv[0, 1] * yy + inv[0, 2]
        sy = inv[1, 0] * pre_x + inv[1, 1] * yy + inv[1, 2]
        xi, yi = np.floor(sx).astype(np.int32), np.floor(sy).astype(np.int32)
        fx, fy = sx - xi, sy - yi
        rows, cols, vals = [], [], []
        for ox, oy, wt in (
            (0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
            (0, 1, (1-fx)*fy), (1, 1, fx*fy),
        ):
            u, v = xi+ox, yi+oy
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (wt > 1e-12)
            rows.append(indices[ok.ravel()])
            cols.append((v[ok] * w + u[ok]).astype(np.int32))
            vals.append(wt[ok])
        out.append(sparse.coo_matrix((np.concatenate(vals),
                                      (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(h*w, h*w)).tocsr())
    return tuple(out)


def forward_approx(flat: np.ndarray, view: View, mats, h: int, w: int):
    """Return predicted y and *active* linear measurement matrix."""
    a = 1.0 - flat if view.kind == "inverted" else flat
    transformed = np.stack([op @ a for op in mats])
    if view.kind == "coded":
        transformed *= mask_bank(h, w, view.n, view.seed).reshape(view.n, -1)
    win = transformed.argmax(axis=0)
    value = transformed[win, np.arange(h*w)]
    if view.kind == "inverted":
        value = 1.0 - value
    mats_out = []
    for k, op in enumerate(mats):
        factor = (win == k).astype(float)
        if view.kind == "coded":
            factor *= mask_bank(h, w, view.n, view.seed)[k].ravel()
        if np.any(factor):
            mats_out.append(sparse.diags(factor, format="csr") @ op)
    active = sum(mats_out, sparse.csr_matrix((h*w, h*w)))
    return value, active


def solve_blind(observations: tuple[np.ndarray, ...], views: tuple[View, ...],
                iterations=90, smoothing=0.0005):
    """One fixed, differentiable blind solver for every arm. No ground truth input."""
    if not views or len(views) != len(observations):
        raise ValueError("view/observation count mismatch")
    arrays = [np.asarray(y, dtype=np.float64) for y in observations]
    h, w = arrays[0].shape
    if any(y.shape != (h,w) for y in arrays):
        raise ValueError("observation shapes differ")
    operators = [shifted_warps(h, w, v.n, v.dx, v.dy) for v in views]
    # ALL preregistered arms start with the same ordinary N16 observation.
    x0 = np.clip(arrays[0], 0, 1).ravel()
    count = len(views)

    def f_and_g(x):
        im = x.reshape(h, w)
        gradient = np.zeros_like(x)
        objective = 0.0
        for y, view, matrices in zip(arrays, views, operators):
            predicted, active = forward_approx(x, view, matrices, h, w)
            error = predicted - y.ravel()
            objective += np.dot(error, error) / (2 * count * h * w)
            gradient += (active.T @ error) / (count * h * w)
        if smoothing:
            sx = im[:, 1:] - im[:, :-1]
            sy = im[1:, :] - im[:-1, :]
            s = smoothing / (h*w)
            objective += s * (np.sum(sx*sx) + np.sum(sy*sy)) / 2
            gg = gradient.reshape(h,w)
            gg[:, 1:] += s*sx
            gg[:, :-1] -= s*sx
            gg[1:, :] += s*sy
            gg[:-1, :] -= s*sy
        return objective, gradient

    r = minimize(f_and_g, x0, jac=True, method="L-BFGS-B",
                 bounds=[(0.0,1.0)] * (h*w),
                 options={"maxiter":iterations, "ftol":1e-12,
                          "gtol":1e-10, "maxls":25})
    return r.x.reshape(h,w), {"iterations": int(r.nit),
                              "success": bool(r.success), "objective": float(r.fun)}


def active_coverage(x: np.ndarray, views: tuple[View, ...]) -> float:
    """First-order source columns connected to at least one winning output."""
    h,w = x.shape
    used = np.zeros(h*w, dtype=bool)
    for view in views:
        mats = shifted_warps(h,w,view.n,view.dx,view.dy)
        _, active = forward_approx(x.ravel(),view,mats,h,w)
        used |= np.asarray((abs(active)>1e-12).sum(axis=0)).ravel()>0
    return float(used.mean())


def witness_differences(views: tuple[View,...], *, size=65):
    a,b = witness(size)
    maxima = [float(np.max(np.abs(observe(a,v)-observe(b,v)))) for v in views]
    return {"max_per_view": maxima, "max_over_views": max(maxima),
            "detected_8bit": bool(max(maxima) > 1/255)}


def run(output="results/coded_v2.json", size=64, iterations=90):
    output = Path(output)
    report = {"protocol":"CODED_PROTOCOL.md", "size":size,
              "iterations":iterations,"smoothing":0.0005,
              "arms":{k:[asdict(v) for v in vs] for k,vs in ARMS.items()},
              "scenes":{}, "collision":{}}
    for name,vs in ARMS.items():
        report["collision"][name] = witness_differences(vs)
        print("collision",name,report["collision"][name],flush=True)
    for scene_name in ("face","symbols","landscape"):
        source = scene(scene_name,size)
        rows = {}
        for name,views in ARMS.items():
            observed = tuple(observe(source,v) for v in views)
            guess,info = solve_blind(observed,views,iterations=iterations)
            fits = [float(psnr(observe(guess,v),y)) for v,y in zip(views,observed)]
            rows[name]={"source_psnr":float(psnr(guess,source)),
                        "forward_psnr_by_view":fits,
                        "min_forward_psnr":min(fits),
                        "source_mae":float(np.mean(abs(guess-source))),
                        "winner_coverage":active_coverage(source,views),
                        "solver":info}
            print(scene_name,name,"source",round(rows[name]["source_psnr"],3),
                  "min-fit",round(min(fits),2),
                  "coverage",round(rows[name]["winner_coverage"],3),
                  flush=True)
        report["scenes"][scene_name]=rows
    keys=("single","duplicate_control","symmetry","center","coded","contrast","max_min")
    report["mean_source_psnr"]={name:float(np.mean([
        report["scenes"][s][name]["source_psnr"] for s in report["scenes"]]))
       for name in keys}
    means=report["mean_source_psnr"]
    report["verdicts"]={
        "reconstruction_gain_vs_symmetry":{k:bool(means[k]>=means["symmetry"]+2)
                                          for k in ("center","coded","contrast")},
        "coded_advantage":bool(means["coded"]>=max(means["center"],means["contrast"])+1),
        "witness_separated":{k:bool(z["detected_8bit"])
                             for k,z in report["collision"].items()}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/coded_v2.json")
    p.add_argument("--size",type=int,default=64)
    p.add_argument("--iterations",type=int,default=90)
    args=p.parse_args()
    r=run(args.output,args.size,args.iterations)
    print("MEANS",json.dumps(r["mean_source_psnr"],sort_keys=True))
    print("VERDICTS",json.dumps(r["verdicts"],sort_keys=True))
    print("WROTE",args.output)


if __name__=="__main__":
    main()
