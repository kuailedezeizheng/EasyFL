class ServerSide(object):
    def __init__(self, global_weight, aggregate_function, root_train_dataset, device, args):
        self.global_weight = global_weight
        self.aggregate_function = aggregate_function
        self.root_train_dataset = root_train_dataset
        self.device = device
        self.args = args

    def aggregate(self, user_model_weight_list):
        self.global_weight = self.aggregate_function(model_weights_list=user_model_weight_list,
                                                     global_model_weights=self.global_weight,
                                                     root_train_dataset=self.root_train_dataset,
                                                     device=self.device,
                                                     args=self.args)

    def get_global_weight(self):
        return self.global_weight
