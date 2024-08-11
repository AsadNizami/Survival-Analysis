import torch
import numpy as np
from lifelines.utils import concordance_index


def cox_loss(prediction, y_true):
    risk = torch.exp(prediction)
    risk_cumsum = torch.cumsum(risk, dim=0)
    log_risk = torch.log(risk_cumsum)
    uncensored_likelihood = prediction - log_risk
    censored_likelihood = uncensored_likelihood * y_true[:, 1]
    neg_likelihood = -torch.sum(censored_likelihood)
    return neg_likelihood


def calculate_cindex(model, data_loader, device):
    model.eval()
    all_preds = []
    all_times = []
    all_events = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs, _ = model(batch_x)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_times.extend(batch_y[:, 0].numpy())
            all_events.extend(batch_y[:, 1].numpy())

    return concordance_index(all_times, -np.array(all_preds), all_events)
