import time
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None
# from flcore.servers.serverbase import Server
# from threading import Thread
#
from servers.serverBase import Server
from clients.clientAvg import clientAVG
from utils.data_utils import set_global_drift_round
import numpy as np

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            drift_interval = getattr(self.args, 'drift_interval', 25)
            if getattr(self.args, 'drift_type', 'none') in ('heavy', 'both') and drift_interval > 0:
                if i > 0 and i % drift_interval == 0:
                    print(f"Round {i}: Triggering Heavy Concept Drift!")

            # update global drift round so clients using DriftDataset will change over time
            set_global_drift_round(i)
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"==================== Round {i:03d} start ====================")
                print("\nEvaluate global model")
                # 先评估全局模型
                test_acc_avg, train_loss_avg, client_test_accuracies, client_train_losses = self.evaluate()
                
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
                # 再进行本地训练
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()

            #todo 暂时不用dlg
            # if self.dlg_eval and i%self.dlg_gap == 0:
            #     self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print(f"time cost: {self.Budget[-1]:.2f}")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(f"{sum(self.Budget[1:])/len(self.Budget[1:]):.2f}")

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0 and getattr(self.args, "eval_new_clients", False):
            print("\n-------------Fine tuning new clients-------------")
            self.set_new_clients(clientAVG)
            self.eval_new_clients = True
            print("\nEvaluate fine-tuned new clients")
            self.evaluate()
            self.save_results()

        # 客户端拿到最终的模型后做本地微调，用来作为pFL的baseline
        #todo ft暂时不做
        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientAVG)
        #     print(f"\n-------------Fine tuning round-------------")
        #     print("\nEvaluate new clients")
        #     self.evaluate()
