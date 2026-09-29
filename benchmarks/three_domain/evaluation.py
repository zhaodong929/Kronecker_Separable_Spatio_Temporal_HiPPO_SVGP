"""Common original-scale scoring, separate from paper admission."""
import numpy as np
from scipy.special import ndtr, ndtri

LEVELS=np.linspace(.05,.95,10)


def gaussian_scores(truth,mean,variance):
    y,mu,var=[np.asarray(x,dtype=float) for x in (truth,mean,variance)]
    if y.shape!=mu.shape or y.shape!=var.shape or not y.size or not all(np.isfinite(x).all() for x in [y,mu,var]) or np.any(var<=0):
        raise ValueError('Matching finite nonempty predictions and positive observation variance required')
    error=y-mu;sd=np.sqrt(var);z=error/sd
    crps=sd*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    coverage=np.array([np.mean(np.abs(z)<=ndtri((1+level)/2)) for level in LEVELS])
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
        nlpd=float(np.mean(.5*(np.log(2*np.pi*var)+z*z))),crps=float(np.mean(crps)),
        ece=float(np.mean(np.abs(coverage-LEVELS))),coverage90=float(np.mean(np.abs(z)<=ndtri(.95))),
        levels=LEVELS.tolist(),coverage=coverage.tolist())


def restore_score_scale(scores,scale):
    scale=float(scale)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Positive target scale required')
    result=dict(scores)
    for key in ['rmse','crps']:result[key]=float(scores[key]*scale)
    result['nlpd']=float(scores['nlpd']+np.log(scale))
    return result


def mixture_scores_from_records(truth,mean,records):
    """Use actual decoder-mixture scores; moment variances are not densities."""
    y,mu=np.asarray(truth),np.asarray(mean)
    if y.shape!=mu.shape or y.ndim!=2 or len(records)!=len(y):
        raise ValueError('Complete per-time mixture records required')
    for i,row in enumerate(records):
        if row['step']!=i+1 or not np.allclose(row['levels'],LEVELS,rtol=0,atol=1e-12):
            raise ValueError('Mixture time order or calibration levels mismatch')
        values=np.array([row['nlpd'],row['crps'],row['coverage90'],*row['coverage']])
        if not np.isfinite(values).all():raise ValueError('Nonfinite mixture score')
    coverage=np.mean([row['coverage'] for row in records],axis=0)
    return dict(rmse=float(np.sqrt(np.mean((y-mu)**2))),
        nlpd=float(np.mean([row['nlpd'] for row in records])),
        crps=float(np.mean([row['crps'] for row in records])),
        coverage90=float(np.mean([row['coverage90'] for row in records])),
        ece=float(np.mean(np.abs(coverage-LEVELS))),levels=LEVELS.tolist(),coverage=coverage.tolist())
