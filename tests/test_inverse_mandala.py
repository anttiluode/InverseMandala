import numpy as np
import pytest

from inverse_mandala import (forward_cv2, forward_linear, blind_inverse,
                            oracle_provenance_inverse, psnr, scene, witness)


def test_identity_and_lower_bound():
    x=scene("face",32)
    y, i, stack=forward_cv2(x,1)
    assert np.allclose(y,x,atol=1e-7)
    assert (i == 0).all()
    for n in (3,7,16):
        y,i,stack=forward_cv2(x,n)
        assert np.all(y>=x.astype(np.float32)-1e-7)
        assert i.min()>=0 and i.max()<n
        assert np.array_equal(y, np.take_along_axis(stack,i[None],axis=0)[0])


def test_collision_witness():
    a,b=witness()
    assert np.sum(np.abs(a-b))==1.0
    assert np.max(np.abs(forward_cv2(a,1)[0]-forward_cv2(b,1)[0]))==1.0
    for n in (3,7,16):
        assert np.array_equal(forward_cv2(a,n)[0],forward_cv2(b,n)[0])


def test_sparse_forward_agrees_with_original():
    for n in (1,3,7,16):
        x=scene("landscape",32)
        cv=forward_cv2(x,n)[0]
        approx=forward_linear(x,n)[0]
        assert np.max(abs(cv-approx))<0.025


def test_inverse_does_not_receive_source_and_reduces_forward_error():
    src=scene("symbols",24)
    y=forward_cv2(src,7)[0]
    cand,info=blind_inverse({7:y},iterations=12,smoothing=0.0005)
    assert cand.shape==src.shape
    assert np.isfinite(cand).all()
    assert np.max(abs(cand-src))>0.01
    assert psnr(forward_cv2(cand,7)[0],y)>psnr(forward_cv2(np.zeros_like(src),7)[0],y)


def test_oracle_provenance_has_finite_coverage():
    x=scene("face",24)
    y,p,_=forward_cv2(x,3)
    guess,info=oracle_provenance_inverse({3:y},{3:p},iterations=20)
    assert guess.shape==x.shape
    assert 0<=info["winner_coverage_fraction"]<=1
    assert np.isfinite(guess).all()


def test_validation():
    with pytest.raises(ValueError): forward_cv2(np.ones((3,3)),2)
    with pytest.raises(ValueError): forward_cv2(np.ones((5,5)),0)
    with pytest.raises(ValueError): blind_inverse({})
