import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from attacks.TriggerAttack import poisonous_data
from defenses.LayerDefense import layer_defense
from tools.sampling import mnist_iid, mnist_noniid, cifar_iid
from tools.load_options import load_config
from update import ClientPoint
from models.NetModel import MLP, CNNMnist, CNNCifar
from defenses.FedAvg import fed_avg
from test import test

matplotlib.use('Agg')

if __name__ == '__main__':
    # parse args
    args = load_config()
    args['device'] = torch.device(
        'cuda:0' if torch.cuda.is_available() and args['gpu'] else 'cpu')

    # load dataset and split users
    if args['dataset'] == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            'data/mnist/',
            train=True,
            download=True,
            transform=trans_mnist)
        dataset_test = datasets.MNIST(
            'data/mnist/',
            train=False,
            download=True,
            transform=trans_mnist)
        # sample users
        if args['iid']:
            dict_users = mnist_iid(dataset_train, args['num_users'])
        else:
            dict_users = mnist_noniid(dataset_train, args['num_users'])
    elif args['dataset'] == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            'data/cifar',
            train=True,
            download=True,
            transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            'data/cifar',
            train=False,
            download=True,
            transform=trans_cifar)
        if args['iid']:
            dict_users = cifar_iid(dataset_train, args['num_users'])
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build poisonous dataset
    poisonous_dataset_train = poisonous_data(dataset_train, args['dataset'])

    # build model
    if args['model'] == 'cnn' and args['dataset'] == 'cifar':
        glob_model = CNNCifar(args=args).to(args['device'])
    elif args['model'] == 'cnn' and args['dataset'] == 'mnist':
        glob_model = CNNMnist(args=args).to(args['device'])
    elif args['model'] == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        glob_model = MLP(
            dim_in=len_in,
            dim_hidden=200,
            dim_out=args['num_classes']).to(
            args['device'])
    else:
        exit('Error: unrecognized model')

    glob_model.train()

    # copy weights
    w_glob = glob_model.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    w_locals = []

    # load aggregate function
    if args['aggregate_function'] == 'layer_defense':
        aggregate_function = layer_defense
    elif args['aggregate_function'] == 'fed_avg':
        aggregate_function = fed_avg
    else:
        raise SystemExit("error aggregate function!")

    if args['all_clients']:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args['num_users'])]

    for iter in range(args['epochs']):
        loss_locals = []
        if not args['all_clients']:
            w_locals = []
        normal_users_number = max(int(args['frac'] * args['num_users']), 1)
        chosen_normal_user_list = np.random.choice(
            range(args['num_users']), normal_users_number, replace=False)

        malicious_users_number = max(
            int(args['malicious_user_rate'] * args['num_users']), 1)
        chosen_malicious_user_list = np.random.choice(
            range(args['num_users']), malicious_users_number, replace=False)

        for chosen_user_id in chosen_normal_user_list:
            print("user %d join in train" % chosen_user_id)
            if chosen_user_id not in chosen_malicious_user_list:
                client_model = ClientPoint(
                    args=args,
                    dataset=dataset_train,
                    idx=dict_users[chosen_user_id])
                w, loss = client_model.train(
                    net=copy.deepcopy(glob_model).to(
                        args['device']))
            else:
                print("user %d is malicious user" % chosen_user_id)
                client_model = ClientPoint(
                    args=args,
                    dataset=poisonous_dataset_train,
                    idx=dict_users[chosen_user_id])
                w, loss = client_model.train(
                    net=copy.deepcopy(glob_model).to(
                        args['device']))
            if args['all_clients']:
                w_locals[chosen_user_id] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # aggregate models
        w_global = aggregate_function(w_locals)

        # copy weight to net_glob
        glob_model.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(
        './save/fed_{}_{}_{}_C{}_iid{}.png'.format(
            args['dataset'],
            args['model'],
            args['epochs'],
            args['frac'],
            args['iid']))

    # testing
    glob_model.eval()
    acc_train, loss_train = test(glob_model, dataset_train, args)
    acc_test, loss_test = test(glob_model, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
