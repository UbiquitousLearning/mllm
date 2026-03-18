def show_config() -> None:
    from . import is_mobile_available

    mobile_enabled = str(is_mobile_available()).lower()
    print(f"mllm mobile: {mobile_enabled}")

    # try import mllm_kernel, if true, print mllm_kernel config
    try:
        import mllm_kernel

        print(f"mllm_kernel: {mllm_kernel.__version__}")
    except ImportError:
        print("mllm_kernel: not found")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="pymllm",
        description="pymllm helper commands.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["show-config"],
        help="Run helper command. Use 'show-config' to print config details.",
    )
    args = parser.parse_args()

    if args.command == "show-config":
        show_config()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
