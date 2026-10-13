# InverseMandala — what did symmetry hide?

An inverse-rendering laboratory for [2MandalaTools](https://github.com/anttiluode/2mandalatools)'s original Python mandala operator. **It rotates an image, horizontally flips odd-indexed copies, and takes their pixelwise maximum.** Inversion is not generally unique: the maximum hides the losing values.

[**Interactive explorer**](https://anttiluode.github.io/InverseMandala/) (GitHub Pages, if enabled) · [Original forward code](https://github.com/anttiluode/2mandalatools/blob/main/frequency_mandalas3.py) · [Frozen protocol](PROTOCOL.md)

## What the "argmax provenance map" actually is

For each output pixel, store **which transformed copy won**:

\[M_N(x)(p)=\max_k(T_kx)(p),\qquad P_N(x)(p)=\arg\max_k(T_kx)(p).\]

This is an application-specific winning-layer index, *not* Bayesian MAP estimation, a token transition map, or a physical depth map. Provenance tells us which branch supplied an output but not the losing values. For a natural image, max brightness is not a realistic occlusion depth model. Think **origami + winner-take-all compositing**, not reversible paper folding. A max-pooling neuron or dendritic subunit is an analogy for nonlinear competition, not an assertion about neural morphology.

## Reproduction

Python 3.10+, dependencies NumPy, SciPy, OpenCV, Pillow, pytest:

```bash
pip install -r requirements.txt
python -m pytest -q
python inverse_mandala.py --output results --size 64 --iterations 90
```

This writes `results/summary.json` and a four-column contact sheet (source, composite, blind inverse, privileged-provenance inverse). Synthetic images are generated deterministically; no dataset, model weights, or internet access is required. The solver **only** receives mandala images. The oracle diagnostic additionally receives the genuine winning-copy map; both candidates are then independently assessed against the concealed source and re-rendered through the exact OpenCV forward law.

Static site: open `index.html` in a modern browser, or serve this directory as GitHub Pages. Load your own image (processed locally in the browser, grayscale center crop) and set symmetry 1–16. See the source, nonlinear composite, winning-copy provenance, and a browser-computed heuristic inverse. The inverse **never consults the original or provenance**, though both are displayed as ground-truth explanations. Browser interpolation is continuous bilinear; OpenCV uses an approximately matching quantized bilinear lookup table. The browser uses fixed 100-step Adam optimization and the scientific Python receipt uses L-BFGS-B; their scores will differ.

## Fixed pilot results (64×64 synthetic scenes)

| Image | Evidence | Blind source PSNR ↑ | Re-render fit PSNR ↑ |
|---|---|---:|---:|
| Face | N=16 | 10.31 dB | 45.33 dB |
| Face | N=7,16 | 14.15 dB | ≥46.92 dB |
| Face | N=3,7,16 | 18.33 dB | ≥53.93 dB |
| Symbols | N=16 | 6.16 dB | 48.75 dB |
| Symbols | N=3,7,16 | 13.25 dB | ≥52.12 dB |
| Landscape | N=16 | 5.46 dB | 51.87 dB |
| Landscape | N=3,7,16 | 16.10 dB | ≥53.93 dB |

Detailed scores including the oracle-winner linear least-squares diagnostic and source-pixel coverage are in [results/summary.json](results/summary.json). A **high rendering score is not a high source reconstruction score**. The optimizer is heuristic and may be improved; these results are not optimal inverse-recovery bounds. The extra provenance map requires storage and is not free. Its linear least-squares reconstruction is a diagnostic, *not* an upper bound on recoverability: a fixed regularizer can still make it perform worse than blind nonlinear fitting.

### Exact non-identifiability witness

Source A is all ones. Source B differs in one interior pixel, which is zero. With the original OpenCV operation:

- At symmetry 1, the outputs differ by 1 at that pixel.
- At symmetry 3, 7 and 16, the outputs are **exactly identical**, element by element.

Thus even the set of N=3,7,16 composites cannot distinguish those two sources. The inverse ambiguity is structural, not an AI training problem. Changing the kinds of observations (not merely increasing the symmetry) may help.

## Active measurements v2 — Varjoluotain meets the mandala

[Interactive scored comparison](https://anttiluode.github.io/InverseMandala/#coded) · [Protocol committed before results](CODED_PROTOCOL.md) · [Reproducible code](coded_mandala.py) · [Raw all-case results](results/coded_v2.json) · [Independent GitHub Actions run](https://github.com/anttiluode/InverseMandala/actions/runs/38434540767)

In [Varjoluotain](https://github.com/anttiluode/Varjoluotain), an occluder makes different hidden scene pixels leave different signatures on a wall. Here the original nonlinear mandala loses nonwinning overlaps. A *pre-fusion* intervention can make a previously losing transformed copy win in a different observation. This is a digital, layer-controllable experiment, **not** a physical shadow-casting experiment.

The frozen test holds the same inverse algorithm and 64×64 synthetic scenes fixed across equal **three-observation** arms. One- and two-observation references are explicitly separate:

| Observations | Mean blind SOURCE PSNR ↑ | Exact white/one-dark-pixel collision separated? |
| --- | ---: | :---: |
| One ordinary N=16 | 7.37 dB | No |
| Three identical N=16 (null) | 7.37 dB | No |
| N=16 with two shifted rotation centers | 10.87 dB | No |
| N=16 + two fixed coded per-branch masks | 13.73 dB | **Yes**, max difference 0.1625 |
| N=16 + inverted contrast + shifted center | 9.36 dB | **Yes**, max difference 1.0 |
| **Ordinary N=3, 7, 16** | **15.45 dB** | No |
| N=16 + inverted contrast (two views) | 7.95 dB | **Yes**, max difference 1.0 |

**Verdict:** The mask intervention defeats the *particular* exact collision that defeats three ordinary symmetry views. It also improves the fixed solver over one N=16 view. But ordinary complementary symmetry views beat all other tested three-view arms in **mean source recovery**, so the mask intervention is **not** the best reconstruction method in this pilot. The predeclared \`coded_advantage\` condition passed **only relative to shifted centers and mixed contrast**, not relative to ordinary symmetry. The stronger predeclared reconstruction-gain-over-symmetry gates **all failed**.

Individually, source PSNR of symmetries vs coded masks is: face **17.63 vs 16.60 dB**, symbols **13.30 vs 12.56 dB**, landscape **15.43 vs 12.04 dB**. High forward agreement is not identity recovery.

The winner-selected Jacobian's source-column coverage need not track PSNR: contrast views can cover many source pixels but leave poor reconstruction under this solver. This is a useful warning that local nonzero sensitivity is not the same as well-conditioned inversion.

### Where the physical analogy stops

These masks apply independently to digitally generated transformed copies *before* the max, with fixed 8×8 per-layer transmission patterns (seed 11/29). A cardboard plate in a real room cannot ordinarily adjust each virtual rotated copy separately. Inversion also knows the intervention pattern. Negative contrast likewise requires the ability to invert the physical stimulus, not merely photograph the same scene again. These interventions are appropriate for a computational instrument but should not be called experimental periscopy.

All trials are synthetic, noise-free, with a known model; a fixed 90-step L-BFGS-B heuristic can underfit and is not an optimal bound. OpenCV uses quantized bilinear interpolation, while the derivative model uses continuous interpolation. Mask storage / measurement-energy costs and natural-image priors are **not** evaluated. The collision counterexample proves only that a specific ambiguity can be broken, not that the original is uniquely recovered.

Reproduce:

\`\`\`bash
python -m pytest -q tests/test_coded_mandala.py
python coded_mandala.py --size 64 --iterations 90 --output results/coded_v2.json
\`\`\`

## Scientific boundaries

The fixed pilot tests three synthetic 64×64 drawings, not natural images. The full application also performs grayscale conversion, radial Fourier EQ, per-frame normalization and false-color rendering; this repo isolates **only** the mandala mixing operator. Source color cannot be reconstructed once grayscale was applied. OpenCV's interpolation and image clipping also prevent an ideal mathematical "unfolding."

A visually convincing reconstruction can be hallucinated. A genuine inverse must be evaluated against the particular source, and uniqueness requires additional evidence/assumptions. The next scientific gate would compare optimized candidates under *more independent measurements* against simple mathematical reconstruction and real-image priors; a trained generative image model is **not** part of v1.

## Origin

The idea came from [2MandalaTools](https://github.com/anttiluode/2mandalatools): increasing rotational symmetry produced elaborate images while concealing their source. This work isolates the **inverse**: which marks survive in a nonlinear composite, which can be made visible by another observation, and how much of the source no longer has a unique answer.
