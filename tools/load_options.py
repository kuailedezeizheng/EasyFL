import toml


# 读取配置文件
def load_config():
    config_file_path = 'configs/config.toml'
    with open(config_file_path, 'r') as file:
        config = toml.load(file)

    return config['args']
