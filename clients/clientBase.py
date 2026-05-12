import copy
import warnings
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data, build_partner_map_from_swap_spec


class Client(object):
#     """
#     Base class for clients in federated learning.
#     """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot
        # Temporal drift tracking (can be driven from command-line)
        self.current_round = 0
        self.drift_type = getattr(args, 'drift_type', 'none')
        self.drift_every = getattr(args, 'drift_every', 5)
        self.drift_noise_step = getattr(args, 'drift_noise_step', 0.01)
        self.drift_noise_max = getattr(args, 'drift_noise_max', 0.10)
        self.drift_rotation_step = getattr(args, 'drift_rotation_step', 5.0)
        self.drift_interval = getattr(args, 'drift_interval', 25)
        self.drift_swap_clients = getattr(args, 'drift_swap_clients', '')
        self.all_train_data = kwargs.get('all_train_data', None)
        self.all_test_data = kwargs.get('all_test_data', None)
        self.drift_partner_map = build_partner_map_from_swap_spec(self.drift_swap_clients)

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs.get('train_slow', False)
        self.send_slow = kwargs.get('send_slow', False)
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        # If temporal drift requested, wrap with DriftDataset if available
        try:
            from utils.data_utils import DriftDataset
            if getattr(self.args, 'drift_type', 'none') != 'none':
                ds = DriftDataset(
                    train_data,
                    client_id=self.id,
                    drift_type=self.drift_type,
                    drift_every=self.drift_every,
                    noise_step=self.drift_noise_step,
                    noise_max=self.drift_noise_max,
                    rotation_step=self.drift_rotation_step,
                    drift_interval=self.drift_interval,
                    partner_map=self.drift_partner_map,
                    all_client_data=self.all_train_data,
                )
                return DataLoader(ds, batch_size, drop_last=True, shuffle=True)
        except Exception:
            pass
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        try:
            from utils.data_utils import DriftDataset
            if getattr(self.args, 'drift_type', 'none') != 'none':
                ds = DriftDataset(
                    test_data,
                    client_id=self.id,
                    drift_type=self.drift_type,
                    drift_every=self.drift_every,
                    noise_step=self.drift_noise_step,
                    noise_max=self.drift_noise_max,
                    rotation_step=self.drift_rotation_step,
                    drift_interval=self.drift_interval,
                    partner_map=self.drift_partner_map,
                    all_client_data=self.all_test_data,
                )
                return DataLoader(ds, batch_size, drop_last=False, shuffle=True)
        except Exception:
            pass
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def _safe_auc(self, y_true, y_prob):
        y_prob = np.asarray(y_prob)
        if not np.isfinite(y_prob).all():
            warnings.warn(
                f"Client {self.id} produced non-finite scores during AUC evaluation; replacing NaN/Inf with finite values.",
                RuntimeWarning,
            )
            finite_vals = y_prob[np.isfinite(y_prob)]
            if finite_vals.size == 0:
                y_prob = np.zeros_like(y_prob)
            else:
                y_prob = np.nan_to_num(
                    y_prob,
                    nan=0.0,
                    posinf=float(np.max(finite_vals)),
                    neginf=float(np.min(finite_vals)),
                )

        try:
            return metrics.roc_auc_score(y_true, y_prob, average='micro')
        except ValueError as exc:
            warnings.warn(
                f"Client {self.id} AUC computation failed ({exc}); returning 0.0 instead.",
                RuntimeWarning,
            )
            return 0.0
#
#     def clone_model(self, model, target):
#         for param, target_param in zip(model.parameters(), target.parameters()):
#             target_param.data = param.data.clone()
#             # target_param.grad = param.grad.clone()
#
#     def update_parameters(self, model, new_params):
#         for param, new_param in zip(model.parameters(), new_params):
#             param.data = new_param.data.clone()
#
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

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

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = self._safe_auc(y_true, y_prob)

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
#
#     # def get_next_train_batch(self):
#     #     try:
#     #         # Samples a new batch for persionalizing
#     #         (x, y) = next(self.iter_trainloader)
#     #     except StopIteration:
#     #         # restart the generator if the previous generator is exhausted.
#     #         self.iter_trainloader = iter(self.trainloader)
#     #         (x, y) = next(self.iter_trainloader)
#
#     #     if type(x) == type([]):
#     #         x = x[0]
#     #     x = x.to(self.device)
#     #     y = y.to(self.device)
#
#     #     return x, y
#
#     def save_item(self, item, item_name, item_path=None):
#         if item_path == None:
#             item_path = self.save_folder_name
#         if not os.path.exists(item_path):
#             os.makedirs(item_path)
#         torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
#
#     def load_item(self, item_name, item_path=None):
#         if item_path == None:
#             item_path = self.save_folder_name
#         return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
#
#     # @staticmethod
#     # def model_exists():
#     #     return os.path.exists(os.path.join("models", "server" + ".pt"))
