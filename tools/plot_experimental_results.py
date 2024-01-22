import matplotlib.pyplot as plt


def plot_curve(data, y_label, title, filename):
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.xlabel('federated learning epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)


def save_accuracy_plots(args, loss_train, test_set_main_accuracy_list, test_set_backdoor_accuracy_list):
    dataset, model, epochs, frac, iid = args['dataset'], args['model'], args['epochs'], args['frac'], args['iid']

    loss_title = f'Changes in loss value of {model} model \n when it suffers trigger attack on dataset {dataset}'
    main_title = f'Main accuracy of {model} model \n when it suffers trigger attack on dataset {dataset}'
    backdoor_title = f'Backdoor accuracy of {model} \n model when it suffers trigger attack on dataset {dataset}'

    plot_curve(loss_train, 'Train Loss Value', loss_title,
               f'./save/fl_loss_{dataset}_{model}_ep:_{epochs}_frac:_{frac}_iid:_{iid}.png')

    plot_curve(test_set_main_accuracy_list, 'Main Accuracy value', main_title,
               f'./save/fl_ma_{dataset}_{model}_ep:_{epochs}_frac:_{frac}_iid:_{iid}.png')

    plot_curve(test_set_backdoor_accuracy_list, 'Backdoor Accuracy value', backdoor_title,
               f'./save/fl_ba_{dataset}_{model}_ep:_{epochs}_frac:_{frac}_iid:_{iid}.png')
