import matplotlib.pyplot as plt


def plot_curve(data, y_label, title, filename):
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.xlabel('federated learning epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)


def save_accuracy_plots(
        args,
        loss_train,
        test_set_main_accuracy_list,
        test_set_backdoor_accuracy_list):
    dataset, model, attacks, defenses, epochs, frac, iid = args['dataset'], args['model'], args[
        'attack_method'], args['aggregate_function'], args['epochs'], args['frac'], args['iid']

    loss_title = f'Changes in loss value of {model} model \n when it suffers trigger attack on dataset {dataset}'
    main_title = f'Main accuracy of {model} model \n when it suffers trigger attack on dataset {dataset}'
    backdoor_title = f'Backdoor accuracy of {model} \n model when it suffers trigger attack on dataset {dataset}'

    ls_img_save_path = (
        f'./save/fl_ls_{dataset}_{model}_attack:{attacks}'
        f'_defense:{defenses}_ep:{epochs}_frac:{frac}_iid:{iid}.png')
    ma_img_save_path = (
        f'./save/fl_ma_{dataset}_{model}_attack:{attacks}'
        f'_defense:{defenses}_ep:{epochs}_frac:{frac}_iid:{iid}.png')
    ba_img_save_path = (
        f'./save/fl_ba_{dataset}_{model}_attack:{attacks}'
        f'_defense:{defenses}_ep:{epochs}_frac:{frac}_iid:{iid}.png')

    plot_curve(
        loss_train,
        'Train Loss Value',
        loss_title,
        ls_img_save_path)

    plot_curve(
        test_set_main_accuracy_list,
        'Main Accuracy value',
        main_title,
        ma_img_save_path)

    plot_curve(
        test_set_backdoor_accuracy_list,
        'Backdoor Accuracy value',
        backdoor_title,
        ba_img_save_path)
