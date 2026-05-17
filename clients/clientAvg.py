# import copy
# import torch
# import numpy as np
import copy
import time
import torch
from clients.clientBase import Client
from utils.mcfl_utils import sanitize_model_

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()
        last_good_state = copy.deepcopy(self.model.state_dict())

        max_local_epochs = self.local_epochs
#         if self.train_slow:
#             max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # if self.train_slow:
                #     time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                if not torch.isfinite(output).all():
                    self.model.load_state_dict(last_good_state, strict=True)
                    sanitize_model_(self.model)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                loss = self.loss(output, y)

                # Check for NaN loss before backprop
                if not torch.isfinite(loss):
                    self.model.load_state_dict(last_good_state, strict=True)
                    sanitize_model_(self.model)
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                self.optimizer.step()
                sanitize_model_(self.model)
                last_good_state = copy.deepcopy(self.model.state_dict())

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
