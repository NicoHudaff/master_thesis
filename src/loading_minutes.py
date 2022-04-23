import pandas as pd
import ray
from pandarallel import pandarallel
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import json
from typing import Optional
import warnings
from tqdm import tqdm

# don't show warnings
warnings.filterwarnings("ignore")

with open("./config.json") as config_file:
    config = json.load(config_file)

pandarallel.initialize()
ray.init(log_to_driver=False, num_cpus=6)

# create the engine
global conn
conn = create_engine(
    f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
).connect()

# @ray.remote
def get_data_minutes(id: int) -> pd.DataFrame:
    '''
    This function will gather information about the games and will group by more than one minute

    Input:  id: The id of the game
    '''
    # query to select all information about the home and away data
    QUERY = f"""
        SELECT
            h.match_id,
            h.minute,
            h.period,
            h.home_content,
            a.away_content
        FROM
            (
                SELECT
                    match_id,
                    minute,
                    period,
                    content AS home_content
                FROM
                    master_thesis.data_minutes
                WHERE
                    home_away = 'home'
                    AND match_id = '{id}'
            ) h
            INNER JOIN (
                SELECT
                    match_id,
                    minute,
                    period,
                    content AS away_content
                FROM
                    master_thesis.data_minutes
                WHERE
                    home_away = 'away'
                    AND match_id = '{id}'
            ) a ON h.match_id = a.match_id
            AND h.minute = a.minute
            AND h.period = a.period
        """

    # execute the query
    df = pd.read_sql(QUERY, conn)
    
    # put string in dict from content
    for col in ["home_content", "away_content"]:
        df[col] = df[col].apply(lambda x: x.replace("nan", "np.nan"))
        df[col] = df[col].apply(eval)
    
    # unpack the dictionary values
    home_content = pd.json_normalize(df.home_content)
    away_content = pd.json_normalize(df.away_content)
    home_content.columns = [f"{c}_home" for c in home_content.columns]
    away_content.columns = [f"{c}_away" for c in away_content.columns]

    # concat the results
    rel_cols = [c for c in df.columns if c not in ["home_content", "away_content"]]
    df = pd.concat([df[rel_cols], home_content, away_content], axis=1)

    # all columns where data is stored in
    cols = [c for c in df.columns if c not in ["match_id", "minute", "period"]]

    # for various groups (2 minutes in one group until 9 minutes in one group)
    for i in range(2, 10):
        # add information for the grouping
        df[f"group_{i}"] = df.apply(lambda x: int((x["minute"] + ((1 - x["period"]) * 45)) / i), axis=1)

        # cols to group by
        cols_2 = ["match_id", "period", f"group_{i}"]

        # grouping the data
        dff = df.groupby(cols_2)[cols].sum().reset_index(drop=False)

        # put the data in the correct format
        df_res = dff[cols_2].rename({f"group_{i}": "group_no"}, axis=1)
        df_res["content"] = dff[[c for c in dff.columns if c not in cols_2]].to_dict("records")
        df_res["content"] = df_res.content.astype(str)

        # write the data to the database
        df_res.to_sql(f"data_group_{i}", conn, schema="master_thesis", if_exists="append", index=False)


def get_ids(limit: Optional[int] = None) -> list:
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
    ids = pd.read_sql(QUERY, conn).match_id.tolist()

    # if the input indicates limit the output
    if limit:
        return ids[:limit]

    return ids


def main() -> None:
    '''
    The main function calls the data gathering function for all games
    '''
    [get_data_minutes(id) for id in tqdm(get_ids())]
    
    # close the connection
    conn.close()


if __name__ == "__main__":
    main()