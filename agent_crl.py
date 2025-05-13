import torch.nn as nn
import logging


log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


class ClassicalPolicyNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
        )
        log.info("Initialized ClassicalPolicyNet with %d actions", n_actions)

    def forward(self, x):
        output = self.net(x)
        log.debug("Classical network output: %s", output.detach().numpy())
        return output
