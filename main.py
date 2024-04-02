import multiprocessing
import os
import subprocess
import time

import psutil

from server_side import federated_learning
from tools.load_options import load_config


def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"])
        gpu_memory = output.decode("utf-8").strip().split("\n")[0]  # 获取第一行数据（仅针对单个 GPU）
        total, used = map(int, gpu_memory.split(", "))
        return total, used
    except Exception as e:
        print("Error:", e)
        return None


def calculate_gpu_memory_utilization():
    gpu_memory = get_gpu_memory_usage()
    if gpu_memory:
        total, used = gpu_memory
        utilization = used / total
        return utilization
    else:
        return None


def load_experiment_config(dataset):
    """Load experiment configuration based on dataset and model."""
    lab_name_map = {
        '1': "(MNIST, LeNet)",
        '2': "(CIFAR-10, MobileNet)",
        '3': "(CIFAR-100, ResNet-18)",
        "4": "(EMNIST, LeNetEmnist)",
        "5": "(FashionMNIST, LeNet)",
        '6': "(MNIST, MNISTCnn)",
        '7': "(CIFAR-10, CIFAR10Cnn)",
        '8': "(CIFAR-10, DenseNet)",
        '9': "(CIFAR-10, GoogleNet)",
        '10': "(CIFAR-10, VGG13)",
        "11": "(FashionMNIST, FashionCNN)",
    }
    lab_name = lab_name_map.get(dataset)
    if lab_name is None:
        return None
    return load_config(lab_name=lab_name)


def worker(num, config):
    """每个进程将执行的函数"""
    print(f"进程 {num} 正在运行，进程ID为 {os.getpid()}")
    federated_learning(config)
    print(f"进程 {num} 执行完毕")


def write_stop_point(attack_name, defence_name, malicious_user_ratio):
    with open("./stop_log.txt", "w") as file:
        file.write(f"Attack: {attack_name}\n")
        file.write(f"Defence: {defence_name}\n")
        file.write(f"M Ratio: {malicious_user_ratio}\n")


def run_mult_FL(config, attacks, defences, malicious_rates):
    processes = []
    i = 0
    condition = True
    for attack in attacks:
        config['attack_method'] = attack
        for defence in defences:
            config['aggregate_function'] = defence
            for m_ratio in malicious_rates:
                config['malicious_user_rate'] = m_ratio
                gpu_utilization = calculate_gpu_memory_utilization()
                cpu_usage = psutil.cpu_percent(interval=None)
                print(f"当前GPU使用率：{gpu_utilization * 100}%")
                print(f"当前CPU使用率：{cpu_usage}%")
                if gpu_utilization > 0.90:
                    condition = False
                    print("stop!" + attack + defence + str(m_ratio))
                    write_stop_point(attack_name=attack, defence_name=defence, malicious_user_ratio=m_ratio)
                    break
                else:
                    p = multiprocessing.Process(target=worker, args=(i, config.copy()))
                    processes.append(p)
                    p.start()
                    i += 1
                time.sleep(10)
            if not condition:
                break
        if not condition:
            break

    # 等待所有进程结束
    for p in processes:
        p.join()

    print("所有进程已完成")


if __name__ == '__main__':
    lab_config = load_experiment_config("1")
    lab_config['server'] = True

    if lab_config is None:  # 处理实验配置加载失败的情况
        print("实验配置加载失败")
        exit()

    mr = [0.2, 0.4, 0.6, 0.8, 0.9]
    attack_list_a = ['trigger', 'semantic']
    attack_list_b = ['blended', 'sig']
    defence_no_root = ['fed_avg', 'flame']
    defence_root = ['fltrust']
    defence_byzantine = ['krum', 'multikrum']
    defence_small = ['small_flame', 'small_fltrust']
    defence_mean = ['median', 'trimmed_mean']

    run_mult_FL(config=lab_config, attacks=attack_list_a, defences=defence_no_root, malicious_rates=mr)
    run_mult_FL(config=lab_config, attacks=attack_list_b, defences=defence_no_root, malicious_rates=mr)

    run_mult_FL(config=lab_config, attacks=attack_list_a, defences=defence_root, malicious_rates=mr)
    run_mult_FL(config=lab_config, attacks=attack_list_b, defences=defence_root, malicious_rates=mr)

    run_mult_FL(config=lab_config, attacks=attack_list_a, defences=defence_byzantine, malicious_rates=mr)
    run_mult_FL(config=lab_config, attacks=attack_list_b, defences=defence_byzantine, malicious_rates=mr)

    run_mult_FL(config=lab_config, attacks=attack_list_a, defences=defence_small, malicious_rates=mr)
    run_mult_FL(config=lab_config, attacks=attack_list_b, defences=defence_small, malicious_rates=mr)

    run_mult_FL(config=lab_config, attacks=attack_list_a, defences=defence_mean, malicious_rates=mr)
    run_mult_FL(config=lab_config, attacks=attack_list_b, defences=defence_mean, malicious_rates=mr)
