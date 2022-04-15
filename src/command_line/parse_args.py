import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--list-models", action="store_true", help="Models to use. If True lists all models and exists.")
    parser.add_argument(
        "configs",
        nargs="*",
        help="Path to config files. See example config files in configs/ directory."
    )

    # parser.add_argument(
    #     "-d", "--dataset", type=str, help="Which dataset to use. Should be a folder under datasets.", required='--list-models' not in sys.argv
    # )
    # parser.add_argument(
    #     "-r", "--robot", choices=["ackermann", "skid"], help="Type of robot to use.", required='--list-models' not in sys.argv
    # )

    # parser.add_argument(
    #     "-m", "--model", type=str, help="Name of model to use. To see which models are available see --list-models.", required='--list-models' not in sys.argv
    # )
    # # TODO: Organize it nicer?
    # parser.add_argument(
    #     "--delay-steps", type=int, default=1, help="Number of steps controls are delayed by.",
    # )


    args = vars(parser.parse_args()).copy()

    if args["list_models"]:
        from models import get_all_models
        print(
            "Available models are:\n  " + "\n  ".join(get_all_models().keys())
        )
        exit(0)

    if len(args["configs"]) == 0:
        parser.error("No config files given.")

    result = []
    for config in args["configs"]:
        with open(config) as f:
            result.append(yaml.safe_load(f))

    return result
