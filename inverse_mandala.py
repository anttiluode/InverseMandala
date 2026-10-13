"""InverseMandala — forward max-composite, provenance and honest inverse probes.

The original 2mandalatools operator (OpenCV warpAffine + odd-index flip + max)
is reproduced by ``forward_cv2``. Inverse optimization uses continuous bilinear
sparse warps, with every final candidate checked again through forward_cv2.

The solver never reads the source; only a separate scorer can evaluate it.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr


def _gray(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 4:
        raise ValueError("Expected a 2-D grayscale image, at least 4x4")
    if not np.isfinite(a).all():
        raise ValueError("Image contains nonfinite values")
    return a


def forward_cv2(image: np.ndarray, symmetry: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faithfully replay frequency_mandalas3.py:create_mandala, plus argmax.

    Input values are not normalized or recolored; provenance is the FIRST
    winning branch (zero-based). Returns composite, index map, all branch images.
    """
    im = _gray(image).astype(np.float32)
    n = int(symmetry)
    if not 1 <= n <= 64:
        raise ValueError("symmetry must be 1..64")
    h, w = im.shape
    center = (w // 2, h // 2)
    copies = []
    for i in range(n):
        m = cv2.getRotationMatrix2D(center, i * 360.0 / n, 1.0)
        warped = cv2.warpAffine(im, m, (w, h))  # defaults match the original
        if i % 2 == 1:
            warped = cv2.flip(warped, 1)
        copies.append(warped)
    stack = np.stack(copies)
    idx = np.argmax(stack, axis=0).astype(np.uint8)
    return np.max(stack, axis=0), idx, stack


@lru_cache(maxsize=40)
def warp_matrices(height: int, width: int, symmetry: int) -> tuple[sparse.csr_matrix, ...]:
    """Sparse differentiable bilinear approximation of OpenCV's warps.

    OpenCV quantizes interpolation weights to a small lookup table; these
    continuous weights differ slightly. This approximation is independently
    checked against the exact forward renderer after inversion.
    """
    h, w, n = int(height), int(width), int(symmetry)
    yy, xx = np.indices((h, w), dtype=np.float64)
    dest_rows = np.arange(h * w, dtype=np.int32)
    matrices = []
    for i in range(n):
        angle = i * 360.0 / n
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        inv = cv2.invertAffineTransform(m)
        u = (w - 1 - xx) if (i % 2) else xx
        sx = inv[0, 0] * u + inv[0, 1] * yy + inv[0, 2]
        sy = inv[1, 0] * u + inv[1, 1] * yy + inv[1, 2]
        xf, yf = np.floor(sx).astype(np.int32), np.floor(sy).astype(np.int32)
        dx, dy = sx - xf, sy - yf
        rr, cc, vv = [], [], []
        for ox, oy, weights in (
            (0, 0, (1 - dx) * (1 - dy)),
            (1, 0, dx * (1 - dy)),
            (0, 1, (1 - dx) * dy),
            (1, 1, dx * dy),
        ):
            qx, qy = xf + ox, yf + oy
            valid = (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h) & (weights > 1e-12)
            flat = valid.ravel()
            rr.append(dest_rows[flat])
            cc.append((qy[valid] * w + qx[valid]).astype(np.int32))
            vv.append(weights[valid].astype(np.float64))
        mat = sparse.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                                shape=(h * w, h * w)).tocsr()
        matrices.append(mat)
    return tuple(matrices)


def forward_linear(x: np.ndarray, symmetry: int, matrices=None):
    a = _gray(x)
    ws = matrices if matrices is not None else warp_matrices(*a.shape, symmetry)
    stacked = np.stack([(matrix @ a.ravel()).reshape(a.shape) for matrix in ws])
    return stacked.max(axis=0), stacked.argmax(axis=0)


def blind_inverse(observations: dict[int, np.ndarray], *, iterations: int = 90,
                  smoothing: float = 0.0005, start: str = "observed") -> tuple[np.ndarray, dict]:
    """Fit observed max-composites, never using the source or true provenance.

    The inverse is nonconvex and not unique. The output is a feasible-ish
    candidate, not a certified reconstruction of the original image.
    """
    if not observations:
        raise ValueError("At least one observation is required")
    obs = {int(n): _gray(y) for n, y in observations.items()}
    shape = next(iter(obs.values())).shape
    if any(y.shape != shape for y in obs.values()):
        raise ValueError("Observation shapes must match")
    h, w = shape
    mats = {n: warp_matrices(h, w, n) for n in obs}
    if start == "observed":
        x0 = np.minimum.reduce(list(obs.values())).ravel().clip(0, 1)
    elif start == "zeros":
        x0 = np.zeros(h * w)
    else:
        raise ValueError("start must be observed or zeros")

    def f_and_g(flat):
        image = flat.reshape(h, w)
        grad = np.zeros(h * w)
        cost = 0.0
        for n, y in obs.items():
            ws = mats[n]
            layer = np.stack([a @ flat for a in ws])
            win = layer.argmax(axis=0)
            yp = np.max(layer, axis=0)
            err = (yp - y.ravel()) / (h * w * len(obs))
            cost += float(np.dot(yp - y.ravel(), err) / 2)
            for k, a in enumerate(ws):
                selected = np.where(win == k, err, 0.0)
                if np.any(selected):
                    grad += a.T @ selected
        if smoothing > 0:
            difx = image[:, 1:] - image[:, :-1]
            dify = image[1:, :] - image[:-1, :]
            scale = smoothing / (h * w)
            cost += scale * (np.sum(difx ** 2) + np.sum(dify ** 2)) / 2
            gg = grad.reshape(shape)
            gg[:, 1:] += scale * difx
            gg[:, :-1] -= scale * difx
            gg[1:, :] += scale * dify
            gg[:-1, :] -= scale * dify
        return cost, grad

    result = minimize(f_and_g, x0=x0, method="L-BFGS-B", jac=True,
                      bounds=[(0., 1.)] * (h * w),
                      options={"maxiter": int(iterations), "ftol": 1e-12, "gtol": 1e-10,
                               "maxls": 25})
    candidate = result.x.reshape(shape)
    return candidate, {"success": bool(result.success), "iterations": int(result.nit),
                       "optimization_cost": float(result.fun), "message": str(result.message),
                       "start": start, "smoothing": smoothing}


def oracle_provenance_inverse(observations: dict[int, np.ndarray],
                              provenance: dict[int, np.ndarray], *, damping=0.02,
                              iterations=90):
    """Linear least squares from the TRUE winning-copy maps (privileged).

    This is a diagnostic, not a blind inverse. The returned state is not
    guaranteed to satisfy max constraints for every branch.
    """
    mats, rhs = [], []
    h, w = next(iter(observations.values())).shape
    for n, y in observations.items():
        who = np.asarray(provenance[n], dtype=np.int32).ravel()
        ws = warp_matrices(h, w, n)
        if who.shape != (h * w,) or np.any((who < 0) | (who >= n)):
            raise ValueError("Invalid provenance map")
        mat = sum((sparse.diags((who == i).astype(float)) @ a for i, a in enumerate(ws)),
                  sparse.csr_matrix((h * w, h * w)))
        mats.append(mat)
        rhs.append(y.ravel())
    A = sparse.vstack(mats, format="csr")
    x = lsqr(A, np.concatenate(rhs), damp=damping, iter_lim=iterations,
             atol=1e-7, btol=1e-7)[0].reshape((h, w))
    covered = np.asarray((abs(A) > 1e-12).sum(axis=0)).ravel() > 0
    return x.clip(0, 1), {"winner_coverage_fraction": float(covered.mean()),
                          "unobserved_source_pixels": int((~covered).sum()),
                          "damping": damping, "iterations": iterations}


def psnr(x, y) -> float:
    mse = float(np.mean((np.asarray(x) - np.asarray(y)) ** 2))
    return float("inf") if mse == 0 else -10 * np.log10(mse)


def scene(name: str, size=64) -> np.ndarray:
    """Deterministic artificial images. No downloaded training dataset."""
    n = size
    canvas = Image.new("L", (n * 3, n * 3), color=18)
    draw = ImageDraw.Draw(canvas)
    k = 3
    if name == "face":
        draw.ellipse((11*k, 5*k, 51*k, 60*k), fill=175, outline=240, width=2*k)
        draw.ellipse((19*k, 25*k, 27*k, 30*k), fill=35)
        draw.ellipse((36*k, 24*k, 44*k, 29*k), fill=43)
        draw.ellipse((23*k, 26*k, 25*k, 28*k), fill=245)
        draw.arc((22*k, 34*k, 42*k, 52*k), 0, 170, fill=53, width=2*k)
        draw.line([(13*k, 12*k),(23*k, 5*k),(38*k, 7*k)], fill=50, width=3*k)
        draw.ellipse((46*k, 14*k, 50*k, 18*k), fill=250)
    elif name == "symbols":
        draw.rectangle((9*k, 7*k, 21*k, 51*k), fill=190)
        draw.rectangle((12*k, 7*k, 37*k, 18*k), fill=190)
        draw.rectangle((15*k, 27*k, 43*k, 38*k), fill=230)
        draw.ellipse((35*k, 43*k, 52*k, 60*k), fill=150, outline=255, width=3*k)
        draw.polygon([(48*k, 4*k), (55*k, 18*k), (38*k, 17*k)], fill=225)
    elif name == "landscape":
        draw.rectangle((0, 41*k, 64*k, 64*k), fill=86)
        draw.ellipse((39*k, 5*k, 55*k, 21*k), fill=240)
        draw.polygon([(0, 48*k), (23*k, 20*k), (43*k, 48*k)], fill=155)
        draw.rectangle((13*k, 35*k, 17*k, 51*k), fill=60)
        draw.ellipse((5*k, 26*k, 25*k, 40*k), fill=225)
        draw.line([(51*k, 41*k), (49*k, 63*k)], fill=250, width=2*k)
    else:
        raise ValueError("Unknown scene")
    canvas = canvas.resize((n, n), resample=Image.Resampling.LANCZOS)
    return np.asarray(canvas, dtype=np.float64) / 255.


def witness(size=65):
    a = np.ones((size, size), dtype=np.float32)
    b = a.copy()
    b[size // 2 - 12, size // 2 + 8] = 0
    return a, b


def evaluate(output=Path("results"), *, size=64, iterations=90):
    """Execute pilot after protocol freeze; no test-time tuning."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contact = []
    all_cases = []
    for name in ("face", "symbols", "landscape"):
        src = scene(name, size)
        obs = {n: forward_cv2(src, n)[0].astype(float) for n in (3, 7, 16)}
        prov = {n: forward_cv2(src, n)[1] for n in (3, 7, 16)}
        for ns in ((16,), (7, 16), (3, 7, 16)):
            selected = {n: obs[n] for n in ns}
            candidate, info = blind_inverse(selected, iterations=iterations)
            oracle, oracle_info = oracle_provenance_inverse(selected, {n:prov[n] for n in ns})
            fit = {str(n): psnr(forward_cv2(candidate, n)[0], obs[n]) for n in ns}
            or_fit = {str(n): psnr(forward_cv2(oracle, n)[0], obs[n]) for n in ns}
            case = {"image":name,"observations":list(ns),
                    "blind_source_psnr":psnr(candidate,src),
                    "blind_forward_psnr":fit,"blind_info":info,
                    "oracle_source_psnr":psnr(oracle,src),
                    "oracle_forward_psnr":or_fit,"oracle_info":oracle_info}
            all_cases.append(case)
            if name == "face" and ns in ((16,), (3,7,16)):
                for a in (src,obs[16],candidate,oracle):
                    contact.append(Image.fromarray(np.uint8(np.clip(a,0,1)*255)).resize((220,220)))
    x, xx = witness()
    diff = {str(n):float(np.max(np.abs(forward_cv2(x,n)[0]-forward_cv2(xx,n)[0])))
            for n in (1,3,7,16)}
    report = {"protocol":"PROTOCOL.md", "source":"2mandalatools/frequency_mandalas3.py:create_mandala",
              "image_size":size,"iterations":iterations,"witness_max_abs_differences":diff,
              "cases":all_cases}
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    if contact:
        cols = 4
        sheet = Image.new("RGB",(cols*220, 2*220))
        for i,a in enumerate(contact): sheet.paste(a.convert("RGB"),((i%cols)*220,(i//cols)*220))
        sheet.save(output/"contact_sheet.png")
    return report


def main():
    p=argparse.ArgumentParser(description="InverseMandala: test loss and reconstructability of nonlinear symmetry composites")
    p.add_argument("--output",default="results")
    p.add_argument("--size",type=int,default=64)
    p.add_argument("--iterations",type=int,default=90)
    args=p.parse_args()
    out=evaluate(Path(args.output),size=args.size,iterations=args.iterations)
    print("Collision witness max |difference|:",out["witness_max_abs_differences"])
    for c in out["cases"]:
        print(f"{c['image']:10} {str(c['observations']):12} blind: source={c['blind_source_psnr']:.2f}dB, forward={min(c['blind_forward_psnr'].values()):.2f}dB; oracle source={c['oracle_source_psnr']:.2f}dB")
    print("Saved",args.output)

if __name__=="__main__": main()
