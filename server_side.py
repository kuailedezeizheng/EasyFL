import torch
from torch.utils.data import DataLoader

from tools.timetamp import add_timestamp
from user_side import UserSide


class ServerSide(object):
    def __init__(self, global_weight, aggregate_function, root_train_dataset, device, model):
        self.global_weight = global_weight
        self.aggregate_function = aggregate_function
        self.device = device
        if aggregate_function.__name__ == 'fltrust':
            self.root_train_loader = DataLoader(root_train_dataset, batch_size=10, drop_last=True, shuffle=True)
            self.root_device = UserSide(model=model, device=device, verbose=False)
        else:
            self.root_train_loader = None
            self.root_device = None

    def aggregate(self, user_model_weight_list):
        if self.root_device is not None:
            self.root_device.reinitialize(model_weight=self.global_weight,
                                          train_dataset_loader=self.root_train_loader,
                                          local_ep=3,
                                          lr=0.01,
                                          batch_size=10)

            self.global_weight = self.aggregate_function(model_weights_list=user_model_weight_list,
                                                         global_model_weights=self.global_weight,
                                                         root_device=self.root_device,
                                                         device=self.device)
        else:
            self.global_weight = self.aggregate_function(model_weights_list=user_model_weight_list,
                                                         global_model_weights=self.global_weight,
                                                         device=self.device)

    def get_global_weight(self):
        return self.global_weight

    def load_model_params(self, filepath):
        self.global_weight.load_state_dict(torch.load(filepath))
