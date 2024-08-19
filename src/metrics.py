import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lifelines.utils import concordance_index


def negative_partial_log(prediction, y_true):
    risk = torch.exp(prediction)
    risk_cumsum = torch.cumsum(risk, dim=0)
    log_risk = torch.log(risk_cumsum)
    uncensored_likelihood = prediction - log_risk
    censored_likelihood = uncensored_likelihood * y_true[:, 1]
    neg_likelihood = -torch.sum(censored_likelihood)
    return neg_likelihood

def ranking_loss(prediction, y_true):
    device = prediction.device
    time = y_true[:, 0].to(device)
    event = y_true[:, 1].to(device)
    
    n_samples = len(prediction)
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for i in range(n_samples):
        if event[i] == 1:  
            longer_survival = time > time[i]
            if longer_survival.sum() > 0:
                diff = prediction[i] - prediction[longer_survival]
                pair_loss = torch.log1p(torch.exp(-diff))
                loss = loss + pair_loss.mean()
    
    if n_samples > 0:
        loss = loss / n_samples
    
    return loss

def combined_cox_ranking_loss(prediction, y_true, alpha=0.8):
    device = prediction.device
    y_true = y_true.to(device)
    
    cox_loss_value = negative_partial_log(prediction, y_true)
    ranking_loss_value = ranking_loss(prediction, y_true)
    
    combined_loss = alpha * cox_loss_value + ranking_loss_value
    
    return combined_loss

def calculate_cindex(model, data_loader, device):
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
