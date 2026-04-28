import copy
import time
import numpy as np
import torch
from clients.clientBase import Client
from utils.pfedme_optimizer import pFedMeOptimizer


class clientpFedMe(Client):
    """pFedMe 客户端：支持个性化参数和全局参数的双轨制"""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda
        self.K = args.K
        self.personalized_learning_rate = args.p_learning_rate

        # 保存本地全局参数和个性化参数
        self.local_params = copy.deepcopy(list(self.model.parameters()))
        self.personalized_params = copy.deepcopy(list(self.model.parameters()))

        self.optimizer = pFedMeOptimizer(
            self.model.parameters(), lr=self.personalized_learning_rate, lamda=self.lamda
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma,
        )

    @staticmethod
    def update_parameters(model, new_params):
        """将参数列表应用到模型"""
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def train(self):
        """pFedMe 训练流程：K步个性化优化 + 本地参数更新"""
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # K 次个性化优化步骤
                for _ in range(self.K):
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.personalized_params = self.optimizer.step_pfedme(self.local_params, self.device)

                # 更新本地全局参数
                for new_param, localweight in zip(self.personalized_params, self.local_params):
                    localweight = localweight.to(self.device)
                    localweight.data = localweight.data - self.lamda * self.learning_rate * (
                        localweight.data - new_param.data
                    )

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # 将更新后的本地参数设回模型
        self.update_parameters(self.model, self.local_params)

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def set_parameters(self, model):
        """从全局模型接收参数，同时更新本地和个性化参数"""
        for new_param, old_param, local_param in zip(model.parameters(), self.model.parameters(), self.local_params):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

    def test_metrics_personalized(self):
        """使用个性化参数评估"""
        testloaderfull = self.load_test_data()
        self.update_parameters(self.model, self.personalized_params)
        self.model.eval()

        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        return test_acc, test_num

    def train_metrics_personalized(self):
        """使用个性化参数评估训练集指标"""
        trainloader = self.load_train_data()
        self.update_parameters(self.model, self.personalized_params)
        self.model.eval()

        train_acc = 0
        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                base_loss = self.loss(output, y).item()

                # 计算正则项：|本地参数 - 个性化参数|
                lm = torch.cat([p.data.view(-1) for p in self.local_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.personalized_params], dim=0)
                reg_loss = 0.5 * self.lamda * torch.norm(lm - pm, p=2).item()

                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]
                losses += (base_loss + reg_loss) * y.shape[0]

        return train_acc, losses, train_num

