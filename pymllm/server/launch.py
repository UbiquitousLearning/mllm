from pymllm.engine.launch import Engine
from pymllm.configs.global_config import make_args, read_args


def _prepare_args():
    parser = make_args()
    read_args(parser=parser)


def main():
    _prepare_args()
    engine = Engine()
    engine.launch()


if __name__ == "__main__":
    main()
