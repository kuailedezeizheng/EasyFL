from torch.utils.tensorboard import SummaryWriter
import csv
import matplotlib.pyplot as plt


def initialize_summary_writer(log_dir):
    return SummaryWriter(log_dir)


def plot_line_chart(writer, accuracy, type_str, step):
    # 将准确值和类型字符串写入Tensorboard
    writer.add_scalar(f'{type_str}', accuracy, global_step=step)


def read_csv(path):
    data = []
    with open(path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        # 跳过第一行（表头）
        next(csvreader)
        for row in csvreader:
            # 将第二列和第三列的数据转换为整数
            row[0] = float(row[0])
            row[1] = float(row[1])
            row[2] = float(row[2])
            data.append(row)
    return data


def get_value(path, step_size=20):
    data = read_csv(path)
    # 从第二列到第三列（Python中索引从0开始）
    ma = [row[0] for row in data]
    ba = [row[1] for row in data]
    loss = [row[2] for row in data]

    return choice_data(
        ma, step_size), choice_data(
        ba, step_size), choice_data(
        loss, step_size)


def choice_data(data, step_size):
    c_data = [d for i, d in enumerate(data) if i % step_size == 0]
    return c_data


def draw_indicator_png(
        path_list,
        title_name_list,
        file_name_list,
        step_length):
    ma_list = []
    ba_list = []
    loss_list = []
    y_label_list = ['MA', 'BA', 'Loss']
    styles = ['-', '--', '-.', ':', '-']
    indicator_colors = ['blue', 'black', 'red', 'green', 'yellow']
    indicator_markers = ['o', '^', 's', 'D', 'v', '*', 'x', '+']
    epochs = [i * step_length for i in range(0, int(1000 / step_length))]

    for i, path in enumerate(path_list):
        ma, ba, loss = get_value(path, step_size=step_length)
        ma_list.append(ma)
        ba_list.append(ba)
        loss_list.append(loss)

    for indicator_index, indicator_list in enumerate(
            [ma_list, ba_list, loss_list]):
        for index, indicator in enumerate(indicator_list):
            plt.plot(
                epochs,
                indicator,
                color=indicator_colors[index],
                linestyle=styles[index],
                marker=indicator_markers[index],
                markersize=5,
                linewidth=1,
                alpha=0.5)

        plt.xlabel('Epoch')
        plt.ylabel(y_label_list[indicator_index])
        plt.title(title_name_list[indicator_index])
        plt.savefig(file_name_list[indicator_index])
        plt.show()


def generate_indicator_legend(
        legend_markers,
        legend_colors,
        legend_name,
        save_path):
    # 创建一个空的图例
    fig, ax = plt.subplots()

    # 将每个标记添加到图例中，同时设置 linestyle 为空字符串
    for indx, marker in enumerate(legend_markers):
        ax.plot(
            [],
            [],
            marker=marker,
            linestyle='',
            label=f'{legend_name[indx]}',
            color=legend_colors[indx],
            alpha=0.5)

    # 设置图例的显示样式，横向显示
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(
            0,
            1.15),
        ncol=len(legend_markers),
        frameon=False)

    # 关闭坐标轴显示
    ax.axis('off')
    # 保存图例为 PDF 文件
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    # 显示图例
    plt.show()


if __name__ == '__main__':
    draw_indicator_png(
        path_list=[
            '../result/csv/lenet-mnist-blended-flame-malicious_rate_0.2-epochs_1000-2024-04-01-03_25_17.csv',
            '../result/csv/lenet-mnist-blended-flame-malicious_rate_0.4-epochs_1000-2024-04-01-03_25_27.csv',
            '../result/csv/lenet-mnist-blended-flame-malicious_rate_0.6-epochs_1000-2024-04-01-03_25_37.csv',
            '../result/csv/lenet-mnist-blended-flame-malicious_rate_0.8-epochs_1000-2024-04-01-03_25_47.csv',
            '../result/csv/lenet-mnist-blended-flame-malicious_rate_0.9-epochs_1000-2024-04-01-03_25_57.csv'],
        title_name_list=[
            'MA of Blended Attack Defense in Flame with LeNet on MNIST',
            'BA of Blended Attack Defense in Flame with LeNet on MNIST',
            'LOSS of Blended Attack Defense in Flame with LeNet on MNIST'],
        file_name_list=[
            '../result/plot/lenet_blended_flame_ma.png',
            '../result/plot/lenet_blended_flame_ba.png',
            '../result/plot/lenet_blended_flame_loss.png'],
        step_length=100)

    l_markers = ['o', '^', 's', 'D', 'v']
    mr_list = [0.2, 0.4, 0.6, 0.8, 0.9]
    colors = ['blue', 'black', 'red', 'green', 'yellow']
    l_name = [f'Malicious Rate: {mr}' for mr in mr_list]
    generate_indicator_legend(
        legend_markers=l_markers,
        legend_name=l_name,
        legend_colors=colors,
        save_path='../result/plot/lenet_blended_flame_legend.pdf')

    draw_indicator_png(
        path_list=[
            '../result/csv/lenet-mnist-blended-flame-malicious_rate_0.2-epochs_1000-2024-04-01-03_25_17.csv',
            '../result/csv/lenet-mnist-blended-krum-malicious_rate_0.2-epochs_1000-2024-04-02-02_17_15.csv'],
        title_name_list=[
            'MA of Blended Attack Against Defense in Different with LeNet on MNIST',
            'BA of Blended Attack Against Defense in Different with LeNet on MNIST',
            'LOSS of Blended Attack Against Defense in Different with LeNet on MNIST'],
        file_name_list=[
            '../result/plot/lenet_blended_different_defense_ma.png',
            '../result/plot/lenet_blended_different_defense_ba.png',
            '../result/plot/lenet_blended_different_defense_loss.png'],
        step_length=100)

    defense_markers = ['o', '^']
    defense_colors = ['blue', 'black']
    defense_name = ["flame", "krum"]
    generate_indicator_legend(
        legend_markers=defense_markers,
        legend_name=defense_name,
        legend_colors=defense_colors,
        save_path='../result/plot/blended_different_defense_legend.pdf')
