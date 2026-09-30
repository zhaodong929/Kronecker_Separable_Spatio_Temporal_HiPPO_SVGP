"""Scoped memoization of the official deterministic HiPPO basis for prediction.

The official prediction and matrix operations are unchanged. The basis depends
on the model and frequencies, not on the query chunk. Never retain it across
prediction calls or use this scope for training.
"""
from functools import wraps
import torch


def cache_prediction_basis(reference):
    @wraps(reference)
    def predict(model,frequencies,x,**kwargs):
        if getattr(model,'_scoped_prediction_basis',False):
            return reference(model,frequencies,x,**kwargs)
        original=model.get_Z
        instance_method=model.__dict__.get('get_Z')
        with torch.no_grad():
            basis=original(frequencies)
        def fixed(w):
            if w is not frequencies:
                raise ValueError('Prediction frequencies changed inside basis scope')
            return basis
        model.get_Z=fixed
        model._scoped_prediction_basis=True
        try:
            return reference(model,frequencies,x,**kwargs)
        finally:
            if instance_method is None:
                delattr(model,'get_Z')
            else:
                model.get_Z=instance_method
            delattr(model,'_scoped_prediction_basis')
    return predict
