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


def get_value(path):
    data = read_csv(path)
    # 从第二列到第三列（Python中索引从0开始）
    ma = [row[0] for row in data]
    ba = [row[1] for row in data]
    loss = [row[2] for row in data]

    return ma, ba, loss


if __name__ == '__main__':
    lab1_path = './csv/lenet-mnist-trigger-krum-malicious_rate:0.2-epochs:10-2024-03-30-19:13:30'
    lab2_path = './csv/lenet-mnist-trigger-krum-malicious_rate:0.4-epochs:10-2024-03-30-19:13:41'
    lab3_path = './csv/lenet-mnist-trigger-krum-malicious_rate:0.6-epochs:10-2024-03-30-19:13:52'
    lab4_path = './csv/lenet-mnist-trigger-krum-malicious_rate:0.8-epochs:10-2024-03-30-19:14:03'
    lab5_path = './csv/lenet-mnist-trigger-krum-malicious_rate:0.9-epochs:10-2024-03-30-19:14:14'

    lab1_ma, lab1_ba, lab1_loss = get_value(lab1_path)
    lab2_ma, lab2_ba, lab2_loss = get_value(lab2_path)
    lab3_ma, lab3_ba, lab3_loss = get_value(lab3_path)
    lab4_ma, lab4_ba, lab4_loss = get_value(lab4_path)
    lab5_ma, lab5_ba, lab5_loss = get_value(lab5_path)

    plt.plot(lab1_ma, label='Malicious User Ratio: 0.2', color='blue', linestyle='-')
    plt.plot(lab2_ma, label='Malicious User Ratio: 0.4', color='red', linestyle='--')
    plt.plot(lab3_ma, label='Malicious User Ratio: 0.6', color='green', linestyle=':')
    plt.plot(lab4_ma, label='Malicious User Ratio: 0.8', color='orange', linestyle='-.')
    plt.plot(lab5_ma, label='Malicious User Ratio: 0.9', color='black', linestyle='-', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('MA')
    plt.title('Tigger Attack Defense in Krum with LeNet on MNIST')
    plt.legend(loc='upper left', fontsize='small')
    plt.savefig('./plot/lenet-trigger-krum.png')
    plt.show()
