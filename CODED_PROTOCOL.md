# Frozen protocol — Coded InverseMandala v2 (active measurements)

Frozen on 2026-10-13 before running **any v2 score**. The v1 code/results are frozen; this experiment adds measurements to its known, nonlinear max-of-rotations source. The biological neuron and cardboard-plate analogies are *motivation*, not claims.

## Question

Can **pre-fusion interventions** make original-pixel distinctions observable that the unmasked, high-symmetry mandala erases? How much do additional observation geometries (symmetry, shifted centers, contrast inversion, fixed coded masks) help reconstruct the specific original scene?

## Source and operator

Re-use exactly v1's deterministic 64x64 grayscale `face`, `symbols`, `landscape` scenes and 65x65 all-ones/one-black-pixel collision witness. For all views, `T_{N,k,c}` is the original OpenCV rotation + alternating horizontal reflection with black border, where the rotation center `c` may vary. Ordinary observation `y = max_k T_k x`; coded observation `y = max_k (m_{k,p} T_k x)` with **fixed, source-independent per-branch masks**. Contrast inversion observation `y = 1 - max_k T_k(1-x)`. Masks/centers/negative contrast must be applied *before* max: modifying the final mandala cannot unmix losing contributions.

Masks are deterministic, generated from seeds 11 and 29, using an 8x8 pattern per branch, nearest-neighbor expanded to image size; values are uniform in [0.15,1]. The 8x8 resolution and seeds are fixed. All mask patterns are disclosed to the inverse solver, and generated globally (not chosen for a particular source); **codebook size and illumination costs are excluded from per-image bytes, but not free in a real device**. The inversion is a mathematical counterfactual where one can control individual transformed layers; it is NOT a model of a passive physical shadow plate over an unknown real photograph.

## Fixed arms / measurement counts

Primary **three-observation** comparison, each observation a full HxW image:

- `symmetry`: ordinary N=16, N=7, N=3.
- `center`: ordinary N=16 at default center, shifted center (+6,-4), shifted center (-5,+5), all N=16.
- `coded`: ordinary N=16 plus two coded N=16 views with seeds 11 and 29.
- `contrast`: ordinary N=16; inverted-contrast N=16; shifted-center (+6,-4) ordinary N=16.
- `duplicate_control`: three identical ordinary N=16 views (detect any optimizer benefit from simply repeating an observation).

Secondary reference `single`: ordinary N=16 only, and `max_min`: ordinary N=16 + inverted-contrast N=16 (two unique measurements).

Each arm gets the **same source**, same number of pixels per observation for primary arms and identical reconstruction method. The contrast condition uses a physically different stimulus (source intensity inversion), not an existing photograph passively measured. Report this access requirement.

## Solver / metric

Use v1's regularized L-BFGS-B inverse method extended to known masked and contrast observations, continuous-bilinear sparse warp approximation, fixed initialization from unmodified N=16 observation, 90 iterations, smoothness weight 0.0005, [0,1] box bounds; no per-arm tuning. Compute source PSNR against the *hidden* input only in scoring, and exact OpenCV re-render PSNR for every supplied view separately. A high forward fit does not prove recovered source. Compare per scene, report mean over these three **illustrative scenes**, not a statistical confidence interval. The optimizer is not a global inverse; poor inverse scores do not establish irrecoverability.

Report the exact witness maximum observation differences between white source and one-dark-pixel source for each arm, and a thresholded 8-bit-change indicator: at least one observation difference > 1/255. **V2 success must not be declared** merely because a coded arm separates the witness; negative contrast will trivially separate such a dark-pixel witness in an otherwise uniform bright image.

Report local first-order visibility as fraction of source pixels with nonzero coefficients in the winner-selected Jacobian rows for each observation stack. This is a necessary coverage diagnostic only, not proof of unique recovery. Also report forward visibility on the witness and, if manageable, a small (16x16) rank check as an optional diagnostic clearly distinguished from source image benchmarking.

## Predeclared readings

- `active_separation`: if coded, contrast or center distinguishes the v1 exact collision witness above the 8-bit threshold, the added measurement defeats that *specific* ambiguity. State clearly when it does not; do not extrapolate to unique inversion.
- `reconstruction_gain`: primary mean source PSNR of a method exceeds `symmetry` by >= 2.0 dB over the three synthetic scenes. Only claim this *if observed*; include individual-scene losses.
- `coded_advantage`: coded beats both center and contrast in mean source PSNR by >= 1.0 dB. Otherwise do not claim masks are a superior probe. Any superiority is limited to this fixed artificial fixture.

## Mandatory nulls and cautions

- Same source with N=1 exactly reveals the grayscale processed input, but is explicitly *not* one of the comparison arms.
- There is no identification of real photographs or latent objects here.
- The per-pixel argmax map is not given to the blind decoder. An observed `y` alone cannot determine the losing contributions.
- `1-max(T(1-x))` has border artifacts because warping uses black borders, so do not claim it equals the per-pixel min of `T x` near image edges.
- The local rank of a max operator changes when winner ties switch; tie-related conclusions are local.
- Fixed synthetic cases, noise-free float images, known forward law, computational masks. Any suggested connection to physically realizable occluders must be separately tested against a physical forward model.
- Preserve every outcome, including negative results, in the README and JSON. 
