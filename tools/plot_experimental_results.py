from torch.utils.tensorboard import SummaryWriter
import csv
import matplotlib.pyplot as plt



def initialize_summary_writer():
    return SummaryWriter()


def plot_line_chart(writer, accuracy, type_str, step):
    # 将准确值和类型字符串写入Tensorboard
    writer.add_scalar(f'{type_str}', accuracy, global_step=step)


def read_csv():
    data = []
    with open('./csv/data.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        # 跳过第一行（表头）
        next(csvreader)
        for row in csvreader:
            # 将第二列和第三列的数据转换为整数
            row[1] = float(row[1])
            row[2] = float(row[2])
            data.append(row)
    return data


def get_value():
    data = read_csv()
    # 从第二列到第三列（Python中索引从0开始）
    step = [row[1] for row in data]
    value = [row[2] for row in data]

    return step, value


if __name__ == '__main__':
    steps, values = get_value()
    plt.plot(steps, values)
    plt.xlabel('Epoch')
    plt.ylabel('BA')
    plt.title('LeNet in MNIST')
    plt.show()
