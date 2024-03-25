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
            row[1] = float(row[1])
            row[2] = float(row[2])
            data.append(row)
    return data


def get_value(path):
    data = read_csv(path)
    # 从第二列到第三列（Python中索引从0开始）
    step = [row[1] for row in data]
    value = [row[2] for row in data]

    return step, value


if __name__ == '__main__':
    lab1_path = './csv/data1.csv'
    lab2_path = './csv/data2.csv'
    lab3_path = './csv/data3.csv'
    lab4_path = './csv/data4.csv'

    lab1_steps, lab1_values = get_value(lab1_path)
    lab2_steps, lab2_values = get_value(lab2_path)
    lab3_steps, lab3_values = get_value(lab3_path)
    lab4_steps, lab4_values = get_value(lab4_path)

    plt.plot(lab1_steps, lab1_values, label='Lab 1', color='blue', linestyle='-')
    plt.plot(lab2_steps, lab2_values, label='Lab 2', color='red', linestyle='--')
    plt.plot(lab3_steps, lab3_values, label='Lab 3', color='green', linestyle=':')
    plt.plot(lab4_steps, lab4_values, label='Lab 4', color='orange', linestyle='-.')

    plt.xlabel('Epoch')
    plt.ylabel('BA')
    plt.title('LeNet in MNIST')
    plt.legend(loc='upper right', fontsize='small')

    plt.show()
