import copy
import time
import numpy as np
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None
from servers.serverBase import Server
from clients.clientPerFedAvg import clientPerFedAvg
from utils.data_utils import set_global_drift_round


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
        prev_drift_state = False  # Track previous drift state to detect transitions

        for i in range(self.global_rounds + 1):
            drift_interval = getattr(self.args, 'drift_interval', 25)
            drift_type = getattr(self.args, 'drift_type', 'none')

            set_global_drift_round(i)

            # Only print drift state changes (not every interval)
            if drift_type in ('heavy', 'both') and drift_interval > 0 and i > 0:
                # Calculate current drift state: True if samples are being swapped
                current_drift_state = (i // drift_interval) % 2 == 1
                if current_drift_state != prev_drift_state:
                    if current_drift_state:
                        print(f"Round {i}: Starting Heavy Concept Drift (swapping client data)")
                    else:
                        print(f"Round {i}: Stopping Heavy Concept Drift (reverting to original data)")
                    prev_drift_state = current_drift_state
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"==================== Round {i:03d} start ====================")
                print("\nEvaluate global model with one step update")
                test_acc_avg, train_loss_avg, client_test_accuracies, client_train_losses = self.evaluate_one_step(round_idx=i)
                
                print(f"Round {i:03d} | "
                      f"Averaged Train Loss: {train_loss_avg:.4f} | "
                      f"Averaged Test Accuracy: {test_acc_avg:.4f} | "
                      f"Min Client Test Accuracy: {np.min(client_test_accuracies):.4f} | "
                      f"Max Client Test Accuracy: {np.max(client_test_accuracies):.4f} | "
                      f"Std Client Test Accuracy: {np.std(client_test_accuracies):.4f}")

                if getattr(self.args, 'wandb', False) and wandb is not None:
                    wandb.log({
                        "round": i,
                        "test_acc": test_acc_avg,
                        "train_loss": train_loss_avg,
                        "client_test_acc_min": np.min(client_test_accuracies),
                        "client_test_acc_max": np.max(client_test_accuracies),
                        "client_test_acc_std": np.std(client_test_accuracies),
                    })
                print(f"==================== Round {i:03d} end ====================")

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print(f"time cost: {self.Budget[-1]:.2f}")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(f"{sum(self.Budget[1:]) / len(self.Budget[1:]):.2f}")

        self.save_results()
        self.save_global_model()

    def evaluate_one_step(self, acc=None, loss=None, round_idx=None):
        """使用一步本地优化来评估模型"""
        # 保存所有客户端当前模型
        models_temp = []
        test_acc = 0.0
        train_loss = 0.0
        accs = [0.0 for _ in self.clients]
        try:
            for c in self.clients:
                models_temp.append(copy.deepcopy(c.model))
                c.train_one_step()

            # 评估测试集
            stats = self.test_metrics()

            # 评估训练集
            stats_train = self.train_metrics()

            accs = [a / n if n else 0.0 for a, n in zip(stats[2], stats[1])]
            test_den = max(sum(stats[1]), 1)
            train_den = max(sum(stats_train[1]), 1)
            test_acc = sum(stats[2]) * 1.0 / test_den
            train_loss = sum(stats_train[2]) * 1.0 / train_den
        except Exception as exc:
            import warnings

            warnings.warn(
                f"PerFedAvg one-step evaluation failed at round {round_idx}: {exc}; using safe fallback values.",
                RuntimeWarning,
            )
            accs = [0.0 for _ in self.clients]
        finally:
            for i, c in enumerate(self.clients):
                if i < len(models_temp):
                    c.clone_model(models_temp[i], c.model)


        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        return test_acc, train_loss, accs, []
