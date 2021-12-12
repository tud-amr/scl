import torch

def cv_ade(inputs, targets, last_n=None):
    """Compute the ADE error loss function .

    Args:
        predictions (torch.Tensor()): Trajectory predictions
        targets (torch.Tensor()): Trajectory targets

    Returns:
        torch.Tensor(): Loss
    """
    v0 = inputs['ego_input']
    # defined in term of increments
    N, T, P = targets.size()

    cv_traj = []
    dt = 0.2
    for i in range(1,T):
        cv_traj.append(v0*dt*i)

    cv_traj = torch.cat(cv_traj,dim=2)

    # Including all truncated backprop outputs (also intermediate)

    loss = torch.mean(torch.linalg.norm(targets - cv_traj,dim=2))

    return loss

def cv_fde(inputs, targets, last_n=None):
    """Compute the ADE error loss function .

    Args:
        predictions (torch.Tensor()): Trajectory predictions
        targets (torch.Tensor()): Trajectory targets

    Returns:
        torch.Tensor(): Loss
    """
    v0 = inputs['ego_input']
    # defined in term of increments
    N, T, P = targets.size()

    cv_traj = []
    dt = 0.2
    for i in range(1,T):
        cv_traj.append(v0*dt*i)

    cv_traj = torch.cat(cv_traj,dim=2)

    # Including all truncated backprop outputs (also intermediate)
    #loss = torch.linalg.norm(targets - cv_traj,dim=2)
    loss = torch.mean(torch.linalg.norm(targets - cv_traj,dim=2)[:, -1])

    return loss

def ade(predictions, targets, last_n=None):
    """Compute the ADE error loss function .

    Args:
        predictions (torch.Tensor()): Trajectory predictions
        targets (torch.Tensor()): Trajectory targets

    Returns:
        torch.Tensor(): Loss
    """
    if len(predictions.size()) > 2:
        # Including all truncated backprop outputs (also intermediate)
        N, T, P = targets.size()
        loss = torch.sum(torch.pow(targets - predictions, 2).reshape([N, T, P // 2, -1]), dim=3)
        loss = torch.sum(torch.sqrt(loss)) / (N * T * (P // 2))
    else:
        # Including only the very last output (so the state is well propagated)
        targets = targets[:, -1, :]
        N, P = targets.size()
        loss = torch.sum(torch.pow(targets - predictions, 2).reshape([N, P // 2, -1]), dim=2)
        loss = torch.sum(torch.sqrt(loss)) / (N * (P // 2))

    return loss


def fde(predictions, targets):
    """Compute the FDE error loss function .

    Args:
        predictions (torch.Tensor()): Trajectory predictions
        targets (torch.Tensor()): Trajectory targets

    Returns:
        torch.Tensor(): Loss
    """
    if len(predictions.size()) > 2:
        # Including all truncated backprop outputs (also intermediate)
        N, T, P = targets.size()
        loss = torch.sum(torch.pow(targets - predictions, 2).reshape([N, T, P // 2, -1]), dim=3)
        loss = torch.sum(torch.sqrt(loss[:, :, -1])) / (N * T)
    else:
        # Including only the very last output (so the state is well propagated)
        targets = targets[:, -1, :]
        N, P = targets.size()
        loss = torch.sum(torch.pow(targets - predictions, 2).reshape([N, P // 2, -1]), dim=2)
        loss = torch.sum(torch.sqrt(loss[:, -1])) / N

    return loss
