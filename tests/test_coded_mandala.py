import numpy as np
import pytest

from inverse_mandala import scene, witness
from coded_mandala import (
    View, NORMAL, CODE_A, CODE_B, INVERTED, CENTER_A, ARMS,
    mask_bank, layers_cv2, observe, shifted_warps, forward_approx,
    solve_blind, active_coverage, witness_differences
)


def test_v1_forward_is_unchanged():
    from inverse_mandala import forward_cv2
    src=scene("face",24)
    for n in (1,3,7,16):
        assert np.array_equal(observe(src,View(n=n)),forward_cv2(src,n)[0])


def test_masks_are_stable_and_pre_max():
    source=scene("symbols",24)
    m1=mask_bank(24,24,16,11)
    m2=mask_bank(24,24,16,29)
    assert m1.shape==(16,24,24)
    assert 0.15<=m1.min()<m1.max()<=1.
    assert not np.array_equal(m1,m2)
    masked=layers_cv2(source,CODE_A)
    vanilla=layers_cv2(source,NORMAL)
    assert np.allclose(masked,vanilla*m1)
    assert np.all(observe(source,CODE_A)<=observe(source,NORMAL)+1e-6)


def test_inverted_view_not_inversion_of_output():
    src=scene("face",24)
    a=observe(src,INVERTED)
    b=1-observe(src,NORMAL)
    assert np.any(abs(a-b)>0.01)
    assert np.array_equal(a,1-layers_cv2(src,INVERTED).max(axis=0))


@pytest.mark.parametrize("view",[NORMAL, CENTER_A, CODE_A, INVERTED])
def test_sparse_model_approximates_exact_view(view):
    src=scene("landscape",24)
    mats=shifted_warps(24,24,view.n,view.dx,view.dy)
    y,_=forward_approx(src.ravel(),view,mats,24,24)
    assert np.max(abs(y.reshape(24,24)-observe(src,view)))<0.045


def test_blind_solver_never_receives_source_or_winner():
    src=scene("face",18)
    views=(NORMAL,INVERTED)
    data=tuple(observe(src,v) for v in views)
    candidate,info=solve_blind(data,views,iterations=6)
    assert candidate.shape==src.shape
    assert np.isfinite(candidate).all()
    assert np.max(abs(candidate-src))>0.01
    assert np.isfinite(info["objective"])


def test_witness_detectability_is_recorded():
    a,b=witness(65)
    assert np.array_equal(observe(a,NORMAL),observe(b,NORMAL))
    assert witness_differences(ARMS["single"])["detected_8bit"] is False
    assert witness_differences(ARMS["symmetry"])["detected_8bit"] is False
    assert witness_differences((NORMAL,INVERTED))["detected_8bit"] is True


def test_local_coverage_bounded():
    v=active_coverage(scene("face",18),(NORMAL,CODE_A))
    assert 0 <= v <= 1


def test_invalid_views():
    with pytest.raises(ValueError):
        View(kind="mystery")
    with pytest.raises(ValueError):
        View(n=0)
    with pytest.raises(ValueError):
        View(kind="coded",seed=100)
