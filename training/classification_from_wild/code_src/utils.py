import torch
import torch.nn.functional as F


def ordinal_cost_fn(logits, levels, imp=None):
    val = - torch.sum(F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels),
                     dim=1,
                     )
    # * imp,
    return torch.mean(val)


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features, targets, levels) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse
