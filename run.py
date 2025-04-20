import argparse
import requests
import yaml
import time
import random


def poisson_request_test(lambda_rate, total_requests, config_path='models_config.yaml', base_port=5000):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    base_url = "http://127.0.0.1:{}/{}"

    # 收集所有模型
    all_models = []
    for family, models in config['families'].items():
        for model in models:
            all_models.append(model)

    # 使用固定种子打散模型顺序,确保每次运行顺序一致
    random.seed(42)
    random.shuffle(all_models)

    # 发送指定总数的请求
    for i in range(total_requests):
        # 随机选择一个模型
        model = random.choice(all_models)
        print(f"Sending request {i+1}/{total_requests} to {model['name']}")
        reqs = {"model_name": model['name'],
                "weights_key": model['weights'],
                "local": True
                }
        try:
            requests.post(base_url.format(base_port, "init"))

            resp = requests.post(base_url.format(
                base_port, "run"), json=reqs)
            print(resp.json())
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
        # 等待一个泊松分布间隔
        wait_time = random.expovariate(lambda_rate)
        time.sleep(wait_time)


def main():
    parser = argparse.ArgumentParser(description="基于泊松分布的模型请求测试")
    parser.add_argument('--lambda_rate', type=float,
                        default=1.0, help='泊松分布的lambda值')
    parser.add_argument('--total_requests', type=int,
                        default=100, help='总的请求次数')
    parser.add_argument('--config', type=str,
                        default='models_config.yaml', help='模型配置文件路径')
    parser.add_argument('--port', type=int, default=5000, help='目标服务器端口号')

    args = parser.parse_args()

    poisson_request_test(
        lambda_rate=args.lambda_rate,
        total_requests=args.total_requests,
        config_path=args.config,
        base_port=args.port
    )


if __name__ == "__main__":
    main()
