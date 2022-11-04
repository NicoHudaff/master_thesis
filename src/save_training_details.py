import warnings
import json
import pandas as pd
from datetime import datetime
from sqlalchemy.engine.base import Engine
from sqlalchemy import create_engine

# don't show warnings since statsbom will warn you every time that you are using the free version
warnings.filterwarnings("ignore")

# SQL connection details are stored in the config file
with open("./config.json") as config_file:
    config = json.load(config_file)


def get_details_model_training(minutes: int, strict: bool) -> pd.DataFrame:
    """
    This function will gather all the details of the training of the models
    It will produce one DataFrame each for each dataset

    Input:  minutes:    The amount of minutes that are summarized
            strict:     Boolean value indicating how the dataset was created
    """
    # define the sorce of the log file
    log_file = "./logs/results_"
    file = f"{log_file}{minutes}{'_2' if (not strict) and (minutes != 1) else ''}.log"

    # gather the information
    df = pd.DataFrame({"message": [l[29:-1] for idx, l in enumerate(open(file, "r"))]})

    # retrieve the epoch number
    df["epoch"] = (
        df.message.apply(
            lambda x: int(x[x.find(" : ") + 3 : x.find("/")])
            if x.startswith("epoch : ")
            else None
        )
        .ffill()
        .astype(int)
    )

    # create masks for all different types
    msk_loss = df.message.apply(lambda x: x.startswith("epoch : "))
    msk_test_loss = df.message.apply(lambda x: x.startswith("test loss is "))
    msk_anomaly_loss = df.message.apply(lambda x: x.startswith("anomaly loss is "))
    msk_f1 = df.message.apply(lambda x: x.startswith("The F1 score is "))
    msk_valid_loss = df.message.apply(lambda x: x.startswith("valid loss is "))
    msk_f1_valid = df.message.apply(
        lambda x: x.startswith("The F1 score (validation set) is ")
    )

    # retrieve the information for each of the different types
    train_loss_df = pd.json_normalize(
        df[msk_loss]
        .reset_index()
        .apply(
            lambda x: {
                "train_loss": float(
                    x["message"][
                        x["message"].find(", loss = ")
                        + 9 : x["message"].find(" for model ")
                    ]
                ),
                "model": x["message"][x["message"].find("for model ") + 10 :],
                "epoch": int(
                    x["message"][x["message"].find(" : ") + 3 : x["message"].find("/")]
                ),
                "epochs": int(
                    x["message"][
                        x["message"].find("/") + 1 : x["message"].find(", loss = ")
                    ]
                ),
            },
            axis=1,
        )
    )
    test_loss_df = pd.json_normalize(
        df[msk_test_loss]
        .reset_index()
        .apply(
            lambda x: {
                "test_loss": float(x["message"][13 : x["message"].find(" for model")]),
                "model": x["message"][x["message"].find("for model ") + 10 :],
                "epoch": x["epoch"],
            },
            axis=1,
        )
    )
    anomaly_loss_df = pd.json_normalize(
        df[msk_anomaly_loss]
        .reset_index()
        .apply(
            lambda x: {
                "anomaly_loss": float(
                    x["message"][16 : x["message"].find(" for model")]
                ),
                "model": x["message"][x["message"].find("for model ") + 10 :],
                "epoch": x["epoch"],
            },
            axis=1,
        )
    )
    f1_df = pd.json_normalize(
        df[msk_f1]
        .reset_index()
        .apply(
            lambda x: {
                "f1_test": float(x["message"][16 : x["message"].find(" for model")]),
                "model": x["message"][x["message"].find("for model ") + 10 :],
                "epoch": x["epoch"],
            },
            axis=1,
        )
    )
    valid_loss_df = pd.json_normalize(
        df[msk_valid_loss]
        .reset_index()
        .apply(
            lambda x: {
                "valid_loss": float(x["message"][14 : x["message"].find(" for model")]),
                "model": x["message"][x["message"].find("for model ") + 10 :],
                "epoch": x["epoch"],
            },
            axis=1,
        )
    )
    f1_valid_df = pd.json_normalize(
        df[msk_f1_valid]
        .reset_index()
        .apply(
            lambda x: {
                "valid_f1": float(x["message"][33 : x["message"].find(" for model")]),
                "model": x["message"][x["message"].find("for model ") + 10 :],
                "epoch": x["epoch"],
            },
            axis=1,
        )
    )

    # merge all the information together
    res_df = (
        train_loss_df.merge(test_loss_df, how="outer", on=["model", "epoch"])
        .merge(anomaly_loss_df, how="outer", on=["model", "epoch"])
        .merge(f1_df, how="outer", on=["model", "epoch"])
        .merge(valid_loss_df, how="outer", on=["model"])
        .merge(f1_valid_df, how="outer", on=["model"])
    )

    # merge the exact model information on the DataFrame
    normalized = pd.json_normalize(res_df.model.apply(eval))
    return pd.concat(
        [
            normalized,
            res_df[
                [
                    c
                    for c in res_df.columns
                    if c not in ["model", "epoch", "epochs", "epoch_y"]
                ]
            ].rename({"epoch_x": "epoch"}, axis=1),
        ],
        axis=1,
    )


def write_to_db(df: pd.DataFrame, engine: Engine) -> None:
    """
    This function formats the give DataFrame to a into the database writable df.
    Afterwards it will be written in the database table.

    Input:  df:     The DataFrame that should be saved
            engine: The sqlalchemy engine regarding the database.
    """
    for col in ["outs", "drops"]:
        df[col] = df[col].astype(str)
    # write it into the database
    with engine.connect() as conn:
        df.to_sql(
            "results_training",
            conn,
            schema="master_thesis",
            if_exists="replace",
            index=False,
        )


def main() -> None:
    """
    This function will write all the training details to the Database
    """
    write_to_db(
        pd.concat(
            [get_details_model_training(i, True) for i in range(1, 10)]
            + [get_details_model_training(i, False) for i in range(2, 10)]
        ).reset_index(drop=True),
        create_engine(
            f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
        ),
    )


if __name__ == "__main__":
    main()
