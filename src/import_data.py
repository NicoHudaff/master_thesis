#!/opt/homebrew/anaconda3/envs/master_thesis/bin/python
# -*- coding: utf-8 -*-

from statsbombpy import sb
import pandas as pd
import ray
from pandarallel import pandarallel
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import json
from typing import Optional

import warnings

# don't show warnings since statsbom will warn you every time that you are using the free version
warnings.filterwarnings("ignore")

# SQL connection details are stored in the config file
with open("./config.json") as config_file:
    config = json.load(config_file)

# initialization for parellization
pandarallel.initialize()
ray.init(log_to_driver=False)

# Define the number of Games that should be loaded in one iteration
CHUNKSIZE_GAMES = 10


def get_available_matches(limit: Optional[int] = None) -> list:
    """
    This function gathers all the game ids regarding the games that are available in the free version of statsbomb

    Input:  limit:  The number of maximum number of games.
    Output:         A list containing all the game ids.
    """
    # first of all gather all leagues and seasons of that leagues that have available data
    available_data = (
        sb.competitions().groupby("competition_id").season_id.apply(list).to_dict()
    )

    # loop over all available seasons to gather the game ids.
    # Since it might return an error only partial list comprehension is possible here.
    matches = []
    for k, v in available_data.items():

        try:
            add = [
                sb.matches(competition_id=k, season_id=w).match_id.unique() for w in v
            ]
            matches += add

        except AttributeError:

            for w in v:
                try:
                    matches += [
                        sb.matches(competition_id=k, season_id=w).match_id.unique()
                    ]

                except AttributeError:
                    pass

    res = [item for sublist in matches for item in sublist]

    # return only partial results if wanted
    if limit is not None:
        return res[:limit]

    return res


@ray.remote
def get_match_data(id: str) -> pd.DataFrame:
    """
    This functions loads the events in form of a data frame given a game id.
    For speed purposes the module ray is used to parallelize the loading.

    Input:  id: The id of that game
    Output:     The information regarding that game given in a DataFrame
    """
    return sb.events(match_id=id)


def load_df_abnormal() -> pd.DataFrame:
    """
    This function loads the information regarding the Hoffenheim Bayern game stored locally.

    Output:     The information regarding that game given in a DataFrame
    """
    df_abnormal = pd.read_csv("./data/df_abnormal.csv")

    # Two columns contain lists or dictionaries, but where saved as a string value.
    # Therefore a transformation must be performed
    for col in ["location", "aktion"]:
        df_abnormal[col] = df_abnormal[col].parallel_apply(eval)

    return df_abnormal


def reformat_abnormal_df(df_abnormal: pd.DataFrame) -> pd.DataFrame:
    """
    Since the format of the anomaly df is different compared to the normal df,
    a transformation needs to be performed.
    Based on the different types of plays the column "aktion" contains differently formated dictionaries.
    These dictionaries needs to be flattened into single columns to match the format of the other df.

    Input:  df_abnormal:    The DataFrame in the old format
    Output: df:             This DataFrame is reformated
    """
    # create masks for different types
    msk_block = df_abnormal["type"] == "Block"
    msk_carry = df_abnormal["type"] == "Carries"
    msk_recover = df_abnormal["type"] == "Ball Recovery"
    msk_receipt = df_abnormal["type"] == "Ball Receipt*"
    msk_pass = df_abnormal["type"] == "Pass"
    msk_miscontrol = df_abnormal["type"] == "Miscontrol"

    # for each individual type different information will be returned
    block_df = pd.DataFrame(
        {
            "id": df_abnormal[msk_block].id.values,
            "block_offensive": df_abnormal[msk_block]
            .aktion.parallel_apply(
                lambda x: x.get("offensive") if isinstance(x, dict) else None
            )
            .values,
        }
    )
    carry_df = pd.DataFrame(
        {
            "id": df_abnormal[msk_carry].id.values,
            "end_location": df_abnormal[msk_carry]
            .aktion.parallel_apply(
                lambda x: x.get("end_location") if isinstance(x, dict) else None
            )
            .values,
        }
    )
    recover_df = pd.DataFrame(
        {
            "id": df_abnormal[msk_recover].id.values,
            "ball_recovery_recovery_failure": df_abnormal[msk_recover]
            .aktion.parallel_apply(
                lambda x: x.get("recovery_failure") if isinstance(x, dict) else None
            )
            .values,
            "ball_recovery_offensive": df_abnormal[msk_recover]
            .aktion.parallel_apply(
                lambda x: x.get("recovery_failure") if isinstance(x, dict) else None
            )
            .values,
        }
    )
    receipt_df = pd.DataFrame(
        {
            "id": df_abnormal[msk_receipt].id.values,
            "ball_receipt_outcome": df_abnormal[msk_receipt]
            .aktion.parallel_apply(
                lambda x: x.get("outcome", {}).get("name")
                if isinstance(x, dict)
                else None
            )
            .values,
        }
    )
    pass_df = pd.DataFrame(
        {
            "id": df_abnormal[msk_pass].id.values,
            "end_location": df_abnormal[msk_pass]
            .aktion.parallel_apply(
                lambda x: x.get("end_location") if isinstance(x, dict) else None
            )
            .values,
            "pass_angle": df_abnormal[msk_pass]
            .aktion.parallel_apply(
                lambda x: x.get("angle") if isinstance(x, dict) else None
            )
            .values,
            "pass_body_part": df_abnormal[msk_pass]
            .aktion.parallel_apply(
                lambda x: x.get("body_part", {}).get("name")
                if isinstance(x, dict)
                else None
            )
            .values,
            "pass_height": df_abnormal[msk_pass]
            .aktion.parallel_apply(
                lambda x: x.get("height", {}).get("name")
                if isinstance(x, dict)
                else None
            )
            .values,
            "pass_length": df_abnormal[msk_pass]
            .aktion.parallel_apply(
                lambda x: x.get("length") if isinstance(x, dict) else None
            )
            .values,
            "pass_recipient": df_abnormal[msk_pass]
            .aktion.parallel_apply(
                lambda x: x.get("recipient", {}).get("name")
                if isinstance(x, dict)
                else None
            )
            .values,
            "pass_type": df_abnormal[msk_pass]
            .aktion.parallel_apply(
                lambda x: x.get("type", {}).get("name") if isinstance(x, dict) else None
            )
            .values,
            "pass_backheel": df_abnormal[msk_pass]
            .aktion.parallel_apply(
                lambda x: x.get("backheel") if isinstance(x, dict) else None
            )
            .values,
        }
    )
    miscontrol_df = pd.DataFrame(
        {
            "id": df_abnormal[msk_miscontrol].id.values,
            "miscontrol_aerial_won": [True for i in range(sum(msk_miscontrol))],
        }
    )

    # merge all the dfs onto the old df based on the id of the play
    df = (
        df_abnormal[
            [c for c in df_abnormal.columns if c not in ["aktion", "miscontrol"]]
        ]
        .merge(block_df, on="id", how="left")
        .merge(carry_df, on="id", how="left")
        .merge(recover_df, on="id", how="left")
        .merge(receipt_df, on="id", how="left")
        .merge(pass_df, on="id", how="left")
        .merge(miscontrol_df, on="id", how="left")
    )

    # Since the column end_location exists multiple times this renaming and concating is performed
    df["end_location"] = df.end_location_x.fillna(df.end_location_y)

    # return only relevant columns
    return df[[c for c in df.columns if c not in ["end_location_y", "end_location_x"]]]


def write_to_db(df: pd.DataFrame, engine: Engine) -> None:
    """
    This function formats the give DataFrame to a into the database writable df.
    Afterwards it will be written in the database table.

    Input:  df:     The DataFrame that should be saved
            engine: The sqlalchemy engine regarding the database.
    """
    # filter for columns that should be saved in a dictionary
    dff = df[
        [
            c
            for c in df.columns
            if c
            not in [
                "type",
                "id",
                "match_id",
                "minute",
                "period",
                "second",
                "team",
                "result",
            ]
        ]
    ]

    # reformat the df
    to_db = pd.DataFrame(
        {
            "id": df.id.values,
            "type": df.type.values,
            "match_id": df.match_id.values,
            "minute": df.minute.values,
            "period": df.period.values,
            "second": df.second.values,
            "team": df.team.values,
            "result": df.result.values,
            "content": dff.to_dict("records"),
        }
    )
    to_db["content"] = to_db.content.astype(str)

    # write it into the database
    with engine.connect() as conn:
        to_db.to_sql("raw_data", conn, schema="master_thesis", if_exists="append", index=False)


def main() -> None:
    """
    The main function gathers information regarding the normal and anomaly soccer game
    Under the column "result" it is stored whether an anomaly has occurred or not.
    The two seperate dfs are finally stored in the same postgres database.

    Output:     The final df with all information in the same format
    """
    # create the engine to connect to the postgres Database
    engine = create_engine(
        f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
    )

    # load the anomaly game and mark it as anomaly
    df_abnormal = reformat_abnormal_df(load_df_abnormal())
    df_abnormal["result"] = "anomaly"
    df_abnormal["period"] = 2.0

    # write the resulting df to the database
    write_to_db(df_abnormal, engine)

    # load information about the ids of the games and the number of games and iterations
    matches = get_available_matches()
    len_matches = len(matches)
    iterations = int(len_matches / CHUNKSIZE_GAMES) + 1

    # perform for each chunk loading and saving of the games information
    for i in range(iterations):
        print("\r", f"{i+1}/{iterations} ({round(((i+1)/iterations)*100)}%)", end="")

        res = ray.get(
            [
                get_match_data.remote(id)
                for id in matches[CHUNKSIZE_GAMES * i : CHUNKSIZE_GAMES * (i + 1)]
            ]
        )
        df = pd.concat(res)

        df["result"] = "normal"

        write_to_db(df, engine)


if __name__ == "__main__":
    main()
