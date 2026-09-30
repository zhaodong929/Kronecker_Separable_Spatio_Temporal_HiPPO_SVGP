import numpy as np
import pytest
from benchmarks.three_domain.era5_alignment import legacy_row_indices

@pytest.mark.parametrize('gap',[None,0,291,293,1859])
def test_identifies_single_dropped_legacy_hour(gap):
    source=np.random.default_rng(39).normal(size=1872)
    expected=np.arange(1860)
    if gap is not None:expected[gap:]+=1
    before=source.copy();indices,found=legacy_row_indices(source[expected],source,.001)
    np.testing.assert_array_equal(indices,expected);assert found==gap
    np.testing.assert_array_equal(source,before)

def test_cannot_hide_an_unexplained_source_difference():
    source=np.random.default_rng(40).normal(size=1872);reference=source[:1860].copy();reference[800]+=2
    with pytest.raises(ValueError,match='review source alignment'):legacy_row_indices(reference,source,.001)
