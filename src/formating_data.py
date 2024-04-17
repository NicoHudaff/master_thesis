import pandas as pd
import ray
from pandarallel import pandarallel
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import json
from typing import Optional
from statsbombpy import sb
from math import sqrt
import warnings

# don't show warnings
warnings.filterwarnings("ignore")

with open("./config.json") as config_file:
    config = json.load(config_file)

with open("./detail_instructions.json") as detail_file:
    detail = json.load(detail_file)

pandarallel.initialize()
ray.init(log_to_driver=False, num_cpus=7)


def get_home_away_information() -> pd.DataFrame:
    """
    This function will gather information about the games and its home and away teams

    Output:     A DataFrame with information regarding the games and their home and away teams
    """
    # first of all gather all leagues and seasons of that leagues that have available data
    available_data = (
        sb.competitions().groupby("competition_id").season_id.apply(list).to_dict()
    )

    # loop over all available seasons to gather the game ids in connection with the teams.
    # Since it might return an error only partial list comprehension is possible here.
    matches = []
    for k, v in available_data.items():

        try:
            add = [
                sb.matches(competition_id=k, season_id=w)[
                    ["home_team", "away_team", "match_id"]
                ]
                for w in v
            ]
            matches += add

        except AttributeError:

            for w in v:
                try:
                    matches += [
                        sb.matches(competition_id=k, season_id=w)[
                            ["home_team", "away_team", "match_id"]
                        ]
                    ]

                except AttributeError:
                    pass

    # concat the results
    df = pd.concat(matches)

    # seperate the home and away teams and mark them with home or away
    home = df[["match_id", "home_team"]].rename({"home_team": "team"}, axis=1)
    away = df[["match_id", "away_team"]].rename({"away_team": "team"}, axis=1)
    home["home_away"] = "home"
    away["home_away"] = "away"

    # concat home and away dfs
    return pd.concat([home, away])


def add_anomaly_home_away(x: pd.core.series.Series) -> str:
    """
    This function will add the home and away information for the Hoffenheim - Bayern game
    Hoffenheim is home and Bayern is away
    If it is not the anomaly game the original home_away information will be returned

    Input:  x:  A column of the DataFrame regarding one minute of one game and one team
    Output:     Home or away information regarding that game and that team
    """
    # if anomaly game and Hoffenheim is the game home will be returned
    if str(x["home_away"]) == "nan" and x["team"] == "Hoffenheim":
        return "home"

    # if anomaly game and Bayern is the game away will be returned
    elif str(x["home_away"]) == "nan":
        return "away"

    # else the original information will be returned
    return x["home_away"]


def get_distance(x: pd.core.series.Series) -> float:
    """
    This function returns the distance between two points
    Given the end_location and the location
    If it is not give, None will be returned

    Input:  x:  The row of a DataFrame, where the locations are stored
    Output:     The distance between the points
    """
    # the end Location is in one of these columns
    end_locs = [
        l
        for l in ["end_location", "carry_end_location", "shot_end_location"]
        if l in x.keys()
    ]

    # if no column is available then None can be returned
    if len(end_locs) == 0:
        return None

    # else we take the first entry and if a z coordinate is also available it will be used
    elif len(x[end_locs[0]]) >= 3:
        return sqrt(
            (x[end_locs[0]][0] - x["location"][0]) ** 2
            + (x[end_locs[0]][1] - x["location"][1]) ** 2
            + (x[end_locs[0]][2]) ** 2
        )

    # else w/o a z coordinate
    return sqrt(
        (x[end_locs[0]][0] - x["location"][0]) ** 2
        + (x[end_locs[0]][1] - x["location"][1]) ** 2
    )


def get_details(
    df: pd.DataFrame,
    type_msk: str,
    rel_events: list,
    dist: bool,
    height: bool,
    dummies: list,
    categories: dict,
    params: dict,
    rename: dict,
) -> pd.DataFrame:
    """
    This function will calculate based on the type count the various events
    Based on the type different events are usefull to count
    Eg for a pass distance might be important as well as for a carry,
    but other types like ball recovery this is not available

    Input:  df:         The df containing all single datapoints
            type_msk:   The type for the analysis should be done
            rel_events: A list of all events that might be related to an action of this type
            dist:       Boolean value whether the distance should be calculated based on the locations
            height:     Boolean value whether the height should be calculated based on the locations
            dummies:    A list of columns for which a dummy columns should be added
            categories: A dictionary indicating which cols can put into categories (long, middle, short)
                        And what the threshholds are
            params:     A dictionary indicating how each columns should be counted ("count" or sum)
            rename:     A dictionary which purpose it is to rename some columns in the final df
    """
    # maybe middle first third or last third

    # get all events for the specific type
    msk = df.type == type_msk
    dff = pd.concat(
        [
            pd.json_normalize(df[msk].content).dropna(axis=1, how="all"),
            df[msk].reset_index(drop=True),
        ],
        axis=1,
    )

    if dff.empty or len(dff) == 0:
        return pd.DataFrame(columns=["match_id", "minute", "team", "period"])

    # add the location roughly
    if "location" in dff.columns:
        dff["x_loc"] = dff.location.apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 1 else None
        )
        dff[f"{type_msk.lower()}_middle"] = (dff.x_loc >= 40) & (dff.x_loc < 80)
        dff[f"{type_msk.lower()}_first"] = dff.x_loc < 40
        dff[f"{type_msk.lower()}_third"] = dff.x_loc >= 80

    # calculate how many related events are there
    if "related_events" in dff.columns and rel_events:
        # map the ids of related events w/ the type
        dff = dff.explode("related_events")
        dff = dff.merge(
            df[["id", "type"]], left_on="related_events", right_on="id", how="left"
        )
        params_rel = {
            c: "first"
            for c in dff.columns
            if c not in ["related_events", "id_x", "id_y", "type_y"]
        }
        params_rel["type_y"] = list
        dff = dff.groupby("id_x").agg(params_rel)

        # check how many relations are there for all possible relations
        for t in rel_events:
            dff[f"{type_msk.lower()}_{t.lower()}_rel"] = dff.type_y.apply(
                lambda x: t in x
            )

    # if the input indicates it calculate the distance
    if dist:
        dff[f"{type_msk.lower()}_dist"] = dff.apply(get_distance, axis=1)

    # if the input indicates it calculate the distance
    if height:
        dff[f"{type_msk.lower()}_height"] = dff.shot_end_location.apply(
            lambda x: x[2] if isinstance(x, list) and len(x) >= 3 else None
        )

    # if the input indicates calculate for the columns dummy columns
    if dummies:
        dummy_cols = []

        for col in [d for d in dummies if d in dff.columns]:
            df_dummies = pd.get_dummies(dff[col], prefix=col)
            dummy_cols.extend(df_dummies.columns)
            dff = pd.concat([df_dummies, dff], axis=1)

    # if the input indicates calculate for the columns the number of short, middle and longs
    if categories:
        for col, thresh in {
            k: v for k, v in categories.items() if k in dff.columns
        }.items():
            dff[f"{type_msk.lower()}_short_{col}"] = dff[col].astype(float) < thresh[0]
            dff[f"{type_msk.lower()}_middle_{col}"] = (
                dff[col].astype(float) < thresh[1]
            ) & (dff[col].astype(float) >= thresh[0])
            dff[f"{type_msk.lower()}_long_{col}"] = dff[col].astype(float) >= thresh[1]

    # update the counting for all added columns in the previous processes
    params.update({f"{type_msk.lower()}_{t.lower()}_rel": sum for t in rel_events})
    if dummies:
        params.update({t: sum for t in dummy_cols})
    if categories:
        params.update(
            {
                f"{type_msk.lower()}_{length}_{col}": sum
                for col in categories.keys()
                for length in ["short", "middle", "long"]
            }
        )
    # add this column to count the total numbers
    params.update(
        {
            "player": "count",
            f"{type_msk.lower()}_middle": sum,
            f"{type_msk.lower()}_first": sum,
            f"{type_msk.lower()}_third": sum,
        }
    )
    rename["player"] = f"total_{type_msk.lower()}"

    # only use the cols that are in the df and replace the string "sum" w/ the function sum
    params = {k: v for k, v in params.items() if k in dff.columns}
    params = {k: v if v != "sum" else sum for k, v in params.items()}

    rename = {k: v for k, v in rename.items() if k in dff.columns}

    # count via a groupby
    return (
        dff.groupby(["match_id", "minute", "period", "team"])
        .agg(params)
        .reset_index(drop=False)
        .rename(
            rename,
            axis=1,
        )
    )


def get_ids(engine: Engine, limit: Optional[int] = None) -> list:
    """
    This function gathers all the ids of games
    and limits the output if indicated

    Input:  engine: The engine for the connection to the postgres database
            limit:  The maximum number of games
    Output: ids:    A list of all ids of the games
    """
    # specify the query
    QUERY = "SELECT match_id FROM master_thesis.materialized_view_raw_data_match"
    # connect to the db and get the ids
    with engine.connect() as conn:
        ids = pd.read_sql(QUERY, conn).match_id.tolist()

    # if the input indicates limit the output
    if limit:
        return ids[:limit]

    return ids


@ray.remote
def get_raw_data(id: str) -> pd.DataFrame:
    """
    This function gathers all the data for one game based on the games id
    The raw data is loaded from the postgres database
    and for all types the events will be counted

    Input:  id:     The id to indicate the game
    Output: ret_df: The DataFrame containing all the data
    """
    # specify the query to gather the raw data from the database
    QUERY = f"SELECT * FROM master_thesis.raw_data WHERE match_id={id}"
    # connect to the database and read the data
    with create_engine(
        f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
    ).connect() as conn:
        df = pd.read_sql(QUERY, conn)

    # due to storing strings "nan" should be exchanged w/ np.nan
    df["content"] = df.content.apply(lambda x: x.replace("nan", "np.nan"))
    df["content"] = df.content.apply(eval)

    # Due to a different Type of Carry storage this needs to be exchanged
    df["type"] = df.type.replace("Carries", "Carry")

    # set up the return df w/ all minutes and teams and periods and match Ids
    QUERY = f"""
        SELECT
            m.*,
            t.team
        FROM
        (
                (
                    SELECT
                        MAX(minute) AS max_min,
                        MIN(minute) AS min_min,
                        match_id,
                        period
                    FROM
                        master_thesis.raw_data
                    WHERE
                        match_id={id}
                    GROUP BY
                        period,
                        match_id
                ) m
                LEFT JOIN (
                    SELECT
                        team,
                        match_id
                    FROM
                        master_thesis.raw_data
                    WHERE
                        match_id={id}
                    GROUP BY
                        team,
                        match_id
                ) t ON t.match_id = m.match_id
            )
    """
    with create_engine(
        f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
    ).connect() as conn:
        ret_df = pd.read_sql(QUERY, conn)
    ret_df["minute"] = ret_df.apply(
        lambda x: list(range(int(x["min_min"]), int(x["max_min"]) + 1)), axis=1
    )
    ret_df = ret_df.explode("minute")[
        ["match_id", "minute", "period", "team"]
    ].reset_index(drop=True)

    # for all available types calculate the number of actions
    for t, d in detail.items():
        add = get_details(
            df,
            t,
            d.get("typen"),
            d.get("dist"),
            d.get("height"),
            d.get("dummies"),
            d.get("categories"),
            d.get("params"),
            d.get("rename"),
        )
        if (not add.empty) and (len(add) != 0):
            ret_df = ret_df.merge(
                add,
                on=["match_id", "minute", "period", "team"],
                how="outer",
            )

    return ret_df.reset_index(drop=True)


def get_db_df(config: dict) -> pd.DataFrame:
    """
    This function concats all the information for the single games into one DataFrame

    Input:  config: The config file to connect to the database
    Output: df:     The DataFrame containing all the information
    """
    # create the engine
    engine = create_engine(
        f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
    )
    # concat all the information
    df = pd.concat(ray.get([get_raw_data.remote(id) for id in get_ids(engine)]))

    # replace True with ones and NaNs with 0s
    df = df.replace(True, 1).fillna(0)

    # merge the home and away information onto the df
    df = df.merge(get_home_away_information(), how="left", on=["match_id", "team"])
    df["home_away"] = df.apply(add_anomaly_home_away, axis=1)

    return df


def write_data_to_db(df: pd.DataFrame) -> None:
    """
    This function will format the DataFrame and write it to the table of the Database

    Input:  df: The information that should be written to the DB
    """
    # format the dataframe such that it can be written in the Table of the DB easily
    df_ret = df[["match_id", "minute", "period", "home_away"]]
    df_ret["content"] = df[
        [
            c
            for c in df.columns
            if c not in ["match_id", "minute", "period", "home_away", "team"]
        ]
    ].to_dict("records")
    df_ret["content"] = df_ret.content.astype(str)
    df_ret["period"] = df_ret.period.astype(int)

    # create the engine
    engine = create_engine(
        f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
    )

    # write the information to the table
    with engine.connect() as conn:
        df_ret.to_sql(
            "data_minutes",
            conn,
            schema="master_thesis",
            if_exists="append",
            index=False,
            chunksize=100,
        )


def main() -> None:
    """
    The main function just calls get_db_df and write the information to the database
    """
    write_data_to_db(get_db_df(config))


if __name__ == "__main__":
    main()
