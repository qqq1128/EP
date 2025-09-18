"""Train an algorithm."""
import argparse
import json
import os

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
import time
start_time = time.time()


def main_name(i):
    """Main function."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--algo",
        type=str,
        default=i,
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "mappo",
            "matd3",
            "masac",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="gym",
        choices=[
            "gym",
        ],
        help="Environment name. Choose from: gym.",
    )
    parser.add_argument("--exp_name", type=str, default="installtest", help="Experiment name.")
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding='utf-8') as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

#~~~~~~~~~~~~~

    # start training
    from harl.runners import RUNNER_REGISTRY
    algo_args["train"]["num_env_steps"] = 24*3001
    algo_args["train"]["n_rollout_threads"] = 1
    algo_args["train"]["train_interval"] = algo_args["algo"]["batch_size"] = 256

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    print(args["algo"])
    runner.name = args["algo"]

    runner.run()

    # 保存模型
    runner.save_dir = "C:\\Users\\qianyi\\Desktop\\HARL-main\\examples\\model\\" + args["algo"]
    # 创建文件夹 (然后把指定的算法塞进去)
    os.makedirs(runner.save_dir, exist_ok=True)
    runner.save()
    # 加载模型
    # runner.restore()

    # 评估模型
    # runner.episode = 1
    # runner.eval()

    runner.close()


def main():
    # "hatd3",
    # "matd3",
    choices = [
        #"hasac",
         #"hatd3",
         # "hasac",
         "masac",
         #  "happo",
        # "hatrpo",
        # "maddpg",
        # "matd3",
        # "mappo",
    ]
    for i in choices:
        main_name(i)


if __name__ == "__main__":
    main()
