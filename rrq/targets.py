"""Matched offline training maps; no claim that a shared generator prefix is frozen."""
import torch


def training_map(clean, predicted, corrected, arm, use_predicted):
    if clean.shape != predicted.shape or clean.shape != corrected.shape or clean.ndim != 3:
        raise ValueError('maps must have identical [Q,B,T] shapes')
    if any(x.dtype != torch.long for x in (clean, predicted, corrected)):
        raise ValueError('maps must be int64')
    if use_predicted.shape != (clean.shape[1],) or use_predicted.dtype != torch.bool:
        raise ValueError('one shared boolean mixture choice per sequence is required')
    if not torch.equal(predicted[0], corrected[0]):
        raise ValueError('RRQ must keep the actual predicted prefix')
    if arm == 'clean':
        return clean.clone()
    if arm not in {'exposure', 'rrq'}:
        raise ValueError('unknown training arm')
    replacement = corrected.clone() if arm == 'rrq' else clean.clone()
    replacement[0] = predicted[0]
    # Return the entire map: visible context and labels must use the SAME map.
    return torch.where(use_predicted.to(clean.device)[None, :, None], replacement, clean)
