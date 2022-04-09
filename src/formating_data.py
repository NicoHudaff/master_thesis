from pyparsing import restOfLine
from statsbombpy import sb
import pandas as pd
import ray
from pandarallel import pandarallel
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import json
import numpy as np
from typing import Optional
from math import sqrt

with open("./config.json") as config_file:
    config = json.load(config_file)

pandarallel.initialize()
ray.init(log_to_driver=False, num_cpus=7)


def get_distance(x):
    try:
        return sqrt(
            (x["carry_end_location"][0] - x["location"][0]) ** 2
            + (x["carry_end_location"][1] - x["location"][1]) ** 2
        )
    except:
        try:
            return sqrt(
                (x["end_location"][0] - x["location"][0]) ** 2
                + (x["end_location"][1] - x["location"][1]) ** 2
            )
        except:
            return None


def get_recoveries_detail(df):
    msk = df.type == "Ball Recovery"
    dff = pd.concat(
        [
            pd.json_normalize(df[msk].content).dropna(axis=1, how="all"),
            df[msk].reset_index(drop=True),
        ],
        axis=1,
    )
    if "related_events" in dff.columns:
        dff = dff.explode("related_events")
        dff = dff.merge(
            df[["id", "type"]], left_on="related_events", right_on="id", how="left"
        )
        params = {
            c: "first"
            for c in dff.columns
            if c not in ["related_events", "id_x", "id_y", "type_y"]
        }
        params["type_y"] = list
        dff = dff.groupby("id_x").agg(params)
        dff["ball_recovery_pass_rel"] = dff.type_y.apply(lambda x: "Pass" in x)
        dff["ball_recovery_pressure_rel"] = dff.type_y.apply(lambda x: "Pressure" in x)
    params = {
        "ball_recovery_recovery_failure": sum,
        "ball_recovery_offensive": sum,
        "out": sum,
        "under_pressure": sum,
        "ball_recovery_pass_rel": sum,
        "ball_recovery_pressure_rel": sum,
        "player": "count",
    }

    params = {k: v for k, v in params.items() if k in dff.columns}
         
    rename = {
                "out": "ball_recovery_out",
                "under_pressure": "ball_recovery_under_pressure",
                "player": "total_ball_recovery",
            }

    rename = {k: v for k, v in rename.items() if k in dff.columns}

    return (
        dff.groupby(["match_id", "minute", "team"])
        .agg(params)
        .reset_index(drop=False)
        .rename(
            rename,
            axis=1,
        )
    )


def get_carries_details(df):
    msk = df.type == "Carry"
    dff = pd.concat(
        [
            pd.json_normalize(df[msk].content).dropna(axis=1, how="all"),
            df[msk].reset_index(drop=True),
        ],
        axis=1,
    )
    typen = [
        "Pass",
        "Ball Receipt*",
        "Pressure",
        "Ball Recovery",
        "Miscontrol",
        "Duel",
        "Block",
        "Dispossessed",
        "Dribble",
        "Foul Won",
        "Foul Committed",
        "Dribbled Past",
        "Goal Keeper",
        "Shot",
        "Interception",
        "50/50",
        "Injury Stoppage",
        "Clearance",
    ]
    if "related_events" in dff.columns:
        dff = dff.explode("related_events")
        dff = dff.merge(
            df[["id", "type"]], left_on="related_events", right_on="id", how="left"
        )
        params = {
            c: "first"
            for c in dff.columns
            if c not in ["related_events", "id_x", "id_y", "type_y"]
        }
        params["type_y"] = list
        dff = dff.groupby("id_x").agg(params)
        for t in typen:
            dff[f"carry_{t.lower()}_rel"] = dff.type_y.apply(lambda x: t in x)
    # duration and distance
    dff["dist"] = dff.apply(get_distance, axis=1)
    dff["carry_short_dist"] = dff.dist < 5
    dff["carry_middle_dist"] = (dff.dist < 15) & (dff.dist >= 5)
    dff["carry_long_dist"] = dff.dist >= 15
    dff["carry_short_duration"] = dff.duration.astype(float) < 1
    dff["carry_middle_duration"] = (dff.duration.astype(float) < 5) & (
        dff.duration.astype(float) >= 1
    )
    dff["carry_long_duration"] = dff.duration.astype(float) >= 5
    params = {
        "under_pressure": sum,
        "player": "count",
        "carry_short_duration": sum,
        "carry_long_duration": sum,
        "carry_middle_duration": sum,
        "carry_short_dist": sum,
        "carry_long_dist": sum,
        "carry_middle_dist": sum,
    }
    params.update({f"carry_{t.lower()}_rel": sum for t in typen})

    params = {k: v for k, v in params.items() if k in dff.columns}

    rename = {"under_pressure": "carry_under_pressure", "player": "total_carry"}

    rename = {k: v for k, v in rename.items() if k in dff.columns}
    
    return (
        dff.groupby(["match_id", "minute", "team"])
        .agg(params)
        .reset_index(drop=False)
        .rename(
            rename, axis=1
        )
    )


def get_pass_detail(df):
    msk = df.type == "Pass"
    dff = pd.concat(
        [
            pd.json_normalize(df[msk].content).dropna(axis=1, how="all"),
            df[msk].reset_index(drop=True),
        ],
        axis=1,
    )
    typen = [
        "Clearance",
        "Ball Receipt*",
        "Pressure",
        "Block",
        "Pass",
        "Ball Recovery",
        "Interception",
        "Goal Keeper",
        "Duel",
        "Carry",
        "Foul Committed",
        "Foul Won",
    ]
    if "related_events" in dff.columns:
        dff = dff.explode("related_events")
        dff = dff.merge(
            df[["id", "type"]], left_on="related_events", right_on="id", how="left"
        )
        params = {
            c: "first"
            for c in dff.columns
            if c not in ["related_events", "id_x", "id_y", "type_y"]
        }
        params["type_y"] = list
        dff = dff.groupby("id_x").agg(params)

        for t in typen:
            dff[f"pass_{t.lower()}_rel"] = dff.type_y.apply(lambda x: t in x)
    
    dff["dist"] = dff.apply(get_distance, axis=1)

    body_parts = [
        "Left Foot",
        "Right Foot",
        "Other",
        "Head",
        "Keeper Arm",
        "Drop Kick",
        "No Touch",
    ]
    if "pass_body_part" in dff.columns:
        for t in body_parts:
            dff[f"pass_{t.lower()}_bp"] = dff.pass_body_part == t

    pass_heights = ["Ground Pass", "High Pass", "Low Pass"]
    if "pass_height" in dff.columns:
        for t in pass_heights:
            dff[f"pass_{t.lower()}_height"] = dff.pass_height == t

    pass_types = [
        "Recovery",
        "Interception",
        "Kick Off",
        "Goal Kick",
        "Corner",
        "Throw-in",
        "Free Kick",
    ]
    if "pass_type" in dff.columns:
        for t in pass_types:
            dff[f"pass_{t.lower()}_type"] = dff.pass_type == t

    pass_outcomes = ["Incomplete", "Out", "Pass Offside", "Unknown", "Injury Clearance"]
    if "pass_outcome" in dff.columns:
        for t in pass_outcomes:
            dff[f"pass_{t.lower()}_outcome"] = dff.pass_outcome == t

    pass_techniques = ["Through Ball", "Straight", "Outswinging", "Inswinging"]
    if "pass_technique" in dff.columns:
        for t in pass_techniques:
            dff[f"pass_{t.lower()}_technique"] = dff.pass_technique == t

    if "pass_angle" in dff.columns:
        dff["pass_short_angle"] = dff.pass_angle < -2
        dff["pass_middle_angle"] = (dff.pass_angle < 2) & (dff.pass_angle >= 2)
        dff["pass_long_angle"] = dff.pass_angle >= 2

    dff["pass_short_duration"] = dff.duration.astype(float) < 0.5
    dff["pass_middle_duration"] = (dff.duration.astype(float) < 2) & (
        dff.duration.astype(float) >= 0.5
    )
    dff["pass_long_duration"] = dff.duration.astype(float) >= 2

    if "pass_length" in dff.columns:
        dff["pass_short_length"] = dff.pass_length.astype(float) < 15
        dff["pass_middle_length"] = (dff.pass_length.astype(float) < 30) & (
            dff.pass_length.astype(float) >= 15
        )
        dff["pass_long_length"] = dff.pass_length.astype(float) >= 30

    params = {
        "counterpress": sum,
        "under_pressure": sum,
        "player": "count",
        "pass_short_duration": sum,
        "pass_long_duration": sum,
        "pass_middle_duration": sum,
        "pass_short_angle": sum,
        "pass_long_angle": sum,
        "pass_middle_angle": sum,
        "pass_short_length": sum,
        "pass_long_length": sum,
        "pass_middle_length": sum,
        "pass_backheel": sum,
        "pass_aerial_won": sum,
        "pass_assisted_shot_id": "count",
        "pass_cross": sum,
        "pass_cut_back": sum,
        "pass_miscommunication": sum,
        "pass_shot_assist": sum,
        "pass_switch": sum,
        "pass_deflected": sum,
        "pass_no_touch": sum,
    }
    params.update({f"pass_{t.lower()}_rel": sum for t in typen})
    params.update({f"pass_{t.lower()}_bp": sum for t in body_parts})
    params.update({f"pass_{t.lower()}_height": sum for t in pass_heights})
    params.update({f"pass_{t.lower()}_type": sum for t in pass_types})
    params.update({f"pass_{t.lower()}_outcome": sum for t in pass_outcomes})
    params.update({f"pass_{t.lower()}_technique": sum for t in pass_techniques})

    params = {k: v for k, v in params.items() if k in dff.columns}

    rename = {
                "counterpress": "pass_counterpress",
                "under_pressure": "pass_under_pressure",
                "player": "total_pass",
                "pass_assisted_shot_id": "pass_assist",
            }

    rename = {k: v for k, v in rename.items() if k in dff.columns}

    return (
        dff.groupby(["match_id", "minute", "team"])
        .agg(params)
        .reset_index(drop=False)
        .rename(
            rename,
            axis=1,
        )
    )


def get_ids(engine, limit=None):
    QUERY = "SELECT match_id FROM master_thesis.materialized_view_raw_data_match"
    with engine.connect() as conn:
        ids = pd.read_sql(QUERY, conn).match_id.tolist()
    if limit:
        return ids[:limit]
    return ids


@ray.remote
def get_raw_data(id):
    QUERY = f"SELECT * FROM master_thesis.raw_data WHERE match_id={id}"
    with create_engine(
        f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
    ).connect() as conn:
        df = pd.read_sql(QUERY, conn)
    df["content"] = df.content.apply(lambda x: x.replace("nan", "np.nan"))
    df["content"] = df.content.apply(eval)
    df["type"] = df.type.replace("Carries", "Carry")
    recovery_df = get_recoveries_detail(df)
    carry_df = get_carries_details(df)
    pass_df = get_pass_detail(df)
    return recovery_df.merge(
        carry_df, on=["match_id", "minute", "team"], how="outer"
    ).merge(pass_df, on=["match_id", "minute", "team"], how="outer")


def get_db_df(config):
    engine = create_engine(
        f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
    )
    df = pd.concat(ray.get([get_raw_data.remote(id) for id in get_ids(engine, limit=20)]))
    return df


def main():
    df = get_db_df(config)
    print(df)

if __name__ == "__main__":
    main()