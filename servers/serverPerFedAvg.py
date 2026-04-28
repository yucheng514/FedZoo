import copy
import time
import numpy as np
from servers.serverBase import Server
from clients.clientPerFedAvg import clientPerFedAvg


class serverPerFedAvg(Server):
    """Per-FedAvg 服务器：使用一步微调来评估全局模型"""

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientPerFedAvg)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []

    def train(self):
        """Per-FedAvg 训练循环"""
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model with one step update")
                self.evaluate_one_step()

            # 每个客户端训练两次
            for client in self.selected_clients:
                client.train()
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def evaluate_one_step(self, acc=None, loss=None):
        """使用一步本地优化来评估模型"""
        # 保存所有客户端当前模型
        models_temp = []
        for c in self.clients:
            models_temp.append(copy.deepcopy(c.model))
            c.train_one_step()

        # 评估测试集
        stats = self.test_metrics()
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)

        # 评估训练集
        stats_train = self.train_metrics()
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)

        accs = [a / n for a, n in zip(stats[2], stats[1])]
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))

