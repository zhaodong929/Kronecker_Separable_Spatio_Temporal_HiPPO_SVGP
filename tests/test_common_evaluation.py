import numpy as np
from benchmarks.three_domain.evaluation import gaussian_scores,restore_score_scale,mixture_scores_from_records,LEVELS
from benchmarks.three_domain.metrics import gaussian_mixture_metrics,gaussian_mixture_calibration


def test_gaussian_and_identical_component_mixture_agree_after_restoration():
    y=np.array([-2.,.3,1.]);mean=np.array([0.,.1,.6]);variance=.7
    gaussian=gaussian_scores(y,mean,np.full(3,variance))
    mixture={**gaussian_mixture_metrics(y,mean[None],variance),**gaussian_mixture_calibration(y,mean[None],variance)}
    for key in ['rmse','nlpd','crps','ece']:np.testing.assert_allclose(gaussian[key],mixture[key],rtol=1e-12,atol=1e-12)
    direct=gaussian_scores(y*2+5,mean*2+5,np.full(3,variance*4))
    restored=restore_score_scale(gaussian,2)
    for key in ['rmse','nlpd','crps','ece','coverage90']:np.testing.assert_allclose(direct[key],restored[key],rtol=1e-12,atol=1e-12)


def test_mixture_ece_pools_coverage_instead_of_averaging_absolute_errors():
    rows=[dict(step=i+1,levels=LEVELS.tolist(),coverage=coverage.tolist(),nlpd=1.,crps=.3,coverage90=.9)
          for i,coverage in enumerate([LEVELS-.04,LEVELS+.04])]
    result=mixture_scores_from_records(np.zeros((2,3)),np.zeros((2,3)),rows)
    assert result['ece']<1e-15
