# InverseMandala v1 — protocol frozen before evaluation

Date: 2026-10-13. This file is committed on the experiment branch *before* scoring the benchmark. Source: `anttiluode/2mandalatools/frequency_mandalas3.py`, `create_mandala`.

## Question

How much information about a source image is identifiable after rotated and alternately reflected copies are fused by **pixelwise maximum**? Does additional symmetry observation or knowing the winning-copy index improve reconstruction? Does a plausible inverse necessarily recover the original?

## Forward mapping

For each symmetry N, generate N images: OpenCV rotation about the center by `i * 360/N` degrees with default bilinear interpolation and black borders, then horizontal reflection when `i` is odd. Output is `max_i warped_i`. Provenance `argmax_i` selects first winner in ties. No frequency filter or false-color map is applied in this stage. The source is a grayscale floating-point image after the original program's preprocessing; reconstructing the original RGB photograph is out of scope.

## Fixed tests

- Exact N=1 identity (floating source); all N outputs >= the untransformed source pixelwise.
- Reconstruction of the forward composite from winner values and provenance.
- Explicit collision witness: two different sources whose N>=3 composites agree to float32 precision, for the chosen interior single-pixel deletion on an all-ones image; if exact equality fails for a particular N, report instead of hiding it.
- Show competing inputs and outputs at N=1,3,7,16.
- Report `winner coverage`: percentage of source pixels with nonzero local influence on at least one winning transformed output, for each image/N; *this is not an information-theoretic identifiability proof*.
- Measure source-recovery PSNR and composite-fit PSNR separately whenever an inverse candidate is tested.
- Compare original/source, one-mandala observation, multiple-N observations, and oracle provenance with a fixed, disclosed solver. Oracle provenance is extra information and its storage is not free.

## Demonstration and baselines

Build a dependency-light Python CLI with deterministic synthetic source images, a CPU numeric reconstruction solver, JSON report and contact sheet. Build a static browser explorer with image loading, symmetry control, provenance visualization and a **clearly labeled heuristic candidate reconstruction** (not a guaranteed true inverse). Tests must verify forward geometry, the collision witness and that the renderer never reads a hidden ground truth during inverse reconstruction. Store outputs under `results/`.

Use several deliberately different deterministic images with distinct object silhouettes and fine markings. Nothing is claimed about natural-image accuracy, practical compression, neural architecture or photograph recovery. No trained generative prior is used. Do not call a nearly matching forward rendering an accurate reconstruction unless the original itself matches.

## Nulls and limitations

The max operation is nonlinear and generally non-injective. Even perfect output agreement need not mean source recovery. Additional observations with different N may reduce ambiguity, but the experiment must *measure* it. Provenance may help identify the winner but cannot reconstruct suppressed values. Prior constraints make a particular inverse plausible, never uniquely factual. OpenCV interpolation, zero padding, ties and mirrors are part of the operator; the exact nonidentifiability witness is a stronger claim than any particular optimizer score.

The experiment is a mechanism demonstration, not evidence of a superior image model.
