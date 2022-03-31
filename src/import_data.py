#!/opt/homebrew/anaconda3/envs/master_thesis/bin/python
# -*- coding: utf-8 -*-

from statsbombpy import sb
import pandas as pd
import ray
from pandarallel import pandarallel

import warnings

# don't show warnings since statsbom will warn you every time that you are using the free version
warnings.filterwarnings("ignore")


# initialization for parellization
pandarallel.initialize()
ray.init(log_to_driver=False)


def get_available_matches(limit: int = None) -> list:
    """
    This function gathers all the game ids regarding the games that are available in the free version of statsbomb

    Input:  limit:  The number of maximum number of games.
    Output:         A list containing all the game ids.
    """
    # first of all gather all leagues and seasons of that leagues that have available data
    available_data = (
        sb.competitions()
        .groupby("competition_id")
        .season_id.apply(list)
        .to_dict()
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


def main() -> pd.DataFrame:
    """
    The main function gathers information regarding the normal and anomaly soccer game
    Under the column "result" it is stored whether an anomaly has occurred or not.
    The two seperate dfs are finally concatenated together.

    Output:     The final df with all information in the same format
    """
    # load the anomaly game and mark it as anomaly
    df_abnormal = reformat_abnormal_df(load_df_abnormal())
    df_abnormal["result"] = "anomaly"

    # load the normal games and mark it as normal
    res = ray.get(
        [get_match_data.remote(id) for id in get_available_matches(limit=100)]
    )
    df = pd.concat(res)
    df["result"] = "normal"

    return pd.concat([df, df_abnormal]).reset_index(drop=True)


if __name__ == "__main__":
    main()
