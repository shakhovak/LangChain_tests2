from utils import dialog_rework, generate_task
import pandas as pd
import logging
import json
import argparse

with open("config.json", "r") as f:
    json_config = json.load(f)
TOKEN = json_config["token"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name",
    help="name of generation model to be used",
    nargs="?",
    default="accounts/fireworks/models/llama-v3p1-8b-instruct",
)


def main(model_name, temperature, top_k):
    logger.info("Starting data rework...")
    data = pd.read_csv("data/test_dialogs_dataset - test_dialogs_dataset.csv")
    data["dialog_reworked"] = data["dialog_text"].apply(lambda x: dialog_rework(x))

    logger.info("Starting task generation...")
    data["task"] = data["dialog_reworked"].apply(
        lambda x: generate_task(
            query=x,
            model_name=model_name,
            fireworks_api_key=TOKEN,
            temperature=temperature,
            top_k=top_k
        )
    )
    data.to_csv('data/reworked.csv')
    logger.info("All done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()
    args = parser.parse_args()
    model_name = args.model_name
    main(model_name, 0.5, 50)
