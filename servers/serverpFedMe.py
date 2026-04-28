import copy
import time
import numpy as np
import h5py
from pathlib import Path
from servers.serverBase import Server
from clients.clientpFedMe import clientpFedMe


class serverpFedMe(Server):
    """pFedMe 服务器：支持个性化参数的联邦学习"""

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientpFedMe)

        self.beta = args.beta
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_test_acc_per = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []

    def train(self):
        """pFedMe 训练循环"""
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized model")
                self.evaluate_personalized()

            for client in self.selected_clients:
                client.train()

            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
            self.receive_models()
            self.aggregate_parameters()
            self.beta_aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc_per))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def beta_aggregate_parameters(self):
        """使用 beta 插值全局模型和前一轮参数"""
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    def test_metrics_personalized(self):
        """使用个性化参数评估测试集"""
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_metrics_personalized()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct

    def train_metrics_personalized(self):
        """使用个性化参数评估训练集"""
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            ct, cl, ns = c.train_metrics_personalized()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, losses

    def evaluate_personalized(self):
        """评估个性化模型性能"""
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_acc = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3]) * 1.0 / sum(stats_train[1])

        self.rs_test_acc_per.append(test_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Personalized Train Accuracy: {:.4f}".format(train_acc))
        print("Averaged Personalized Test Accuracy: {:.4f}".format(test_acc))

    def save_results(self):
        """保存个性化学习结果"""
        algo = self.dataset + "_" + self.algorithm
        result_path = Path("../results/")
        result_path.mkdir(parents=True, exist_ok=True)

        if len(self.rs_test_acc_per):
            algo2 = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path / f"{algo2}.h5"
            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("rs_test_acc", data=self.rs_test_acc_per)
                hf.create_dataset("rs_train_acc", data=self.rs_train_acc_per)
                hf.create_dataset("rs_train_loss", data=self.rs_train_loss_per)

