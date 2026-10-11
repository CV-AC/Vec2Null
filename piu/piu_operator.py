import torch


def project_update(updates, bases, gradients, mode='layer', strength=1.0):


    if mode not in ('none', 'nullspace', 'layer', 'global') or not 0 <= strength <= 1:
        raise ValueError('Invalid PIU mode or strength')
    if mode == 'none' or strength == 0:
        return dict(updates)
    selected = [name for name in updates if name in bases]
    if not selected:
        raise ValueError('PIU has no covered update tensors')
    u = {n: bases[n].detach().to(updates[n]) for n in selected}
    coordinates = {n: updates[n] @ u[n] for n in selected}
    result = dict(updates)
    if mode == 'nullspace':
        for n in selected:
            result[n] = updates[n] - strength*(coordinates[n] @ u[n].T)
        return result
    if any(n not in gradients for n in selected):
        raise ValueError('Missing retain-gradient calibration')
    sensitive_grad = {n: gradients[n].detach().to(updates[n]) @ u[n] for n in selected}
    groups = [[n] for n in selected] if mode == 'layer' else [selected]
    for names in groups:
        dot = sum((coordinates[n]*sensitive_grad[n]).sum() for n in names)
        norm = sum(sensitive_grad[n].square().sum() for n in names)
        coefficient = strength*torch.relu(dot)/norm.clamp_min(1e-24)
        for n in names:
            result[n] = updates[n] - coefficient*(sensitive_grad[n] @ u[n].T)
    return result
