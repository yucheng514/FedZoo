import copy
import time
import numpy as np
import torch
from clients.clientBase import Client
from utils.pfedme_optimizer import PerAvgOptimizer


class clientPerFedAvg(Client):
    """Per-FedAvg 客户端：保留全局模型更新能力，同时保持个性化微调"""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.beta = self.learning_rate
        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma,
        )

    def train(self):
        """Per-FedAvg 训练流程：每批次内分两步，第二步使用 beta 回退"""
        trainloader = self.load_train_data(self.batch_size * 2)
        start_time = time.time()
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))

        for _ in range(max_local_epochs):
            for X, Y in trainloader:
                # 保存当前模型参数（第一步之前）
                temp_model = copy.deepcopy(list(self.model.parameters()))

                # 第一步：在前半批数据上更新
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else:
                    x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 第二步：在后半批数据上更新，计算梯度
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device)
                    x[1] = X[1][self.batch_size:]
                else:
                    x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                # 恢复模型参数到第一步之前
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                # 使用 beta 进行加权更新
                self.optimizer.step(beta=self.beta)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def train_one_step(self):
        """单步训练（用于评估时的一步微调）"""
        trainloader = self.load_train_data(self.batch_size)
        x, y = next(iter(trainloader))
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        output = self.model(x)
        loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def clone_model(self, source_model, target_model):
        """复制模型参数"""
        for new_param, old_param in zip(source_model.parameters(), target_model.parameters()):
            old_param.data = new_param.data.clone()

