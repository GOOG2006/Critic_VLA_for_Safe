"""Safety Critic: predicts future collision risk from current state."""
import numpy as np
import torch
import torch.nn as nn


class SafetyCritic(nn.Module):
    """Anticipatory safety critic for collision prediction.

    Input (17-dim):
        - eef_pos (3): end-effector position
        - eef_ori (3): end-effector orientation (axis-angle)
        - gripper (2): gripper joint positions
        - action (7): current nominal action
        - h_analytic (1): current geometric barrier value
        - task_phase (1): bowl lifted indicator

    Output (2-dim):
        - collision_risk_logit: future collision probability (before sigmoid)
        - h_predicted: predicted future minimum h value
    """

    def __init__(self, d_in=17, d_hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_hid), nn.ReLU(),
            nn.Linear(d_hid, 2),
        )

    def forward(self, x):
        out = self.net(x)
        return out[:, 0], out[:, 1]  # risk_logit, h_pred

    def predict_risk(self, state_vec):
        """Convenience: return collision probability from numpy state vector."""
        with torch.no_grad():
            x = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0)
            risk_logit, h_pred = self(x)
            risk_prob = torch.sigmoid(risk_logit).item()
        return risk_prob, h_pred.item()
