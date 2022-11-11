from sqlalchemy import create_engine
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json
import numpy as np
import warnings
from plotly.graph_objs._figure import Figure
from typing import Optional
import  scipy.signal.signaltools

# don't show warnings
warnings.filterwarnings("ignore")

with open("../config.json") as config_file:
    config = json.load(config_file)

# create the engine
conn = create_engine(
    f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
).connect()


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def manipulate(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function will add specific colums to the DataFrame

    Input:  df: The DataFrame w/o certain columns
    Output: df: The DataFrame w/ certain columns
    """
    df["n_layers"] = df.n_layers.astype(str)
    df["model"] = df.apply(lambda x: str({"lr": x["lr"], "opt": x["opt"], "n_layers": x["n_layers"], "outs": x["outs"], "drops": x["drops"], "epochs": x["epochs"]}), axis=1)
    df["dataset"] = df.apply(lambda x: str({"minutes": x["minutes"], "strict": x["strict"]}), axis=1)
    df["epoch_ratio"] = df.epoch / df.epochs
    for c in ["drops", "outs"]:
        df[c] = df[c].apply(eval)

    df["first_layer"] = df.outs.apply(lambda x: x[0])
    df["second_layer"] = df.outs.apply(lambda x: x[1] if len(x) > 1 else None)
    df["third_layer"] = df.outs.apply(lambda x: x[2] if len(x) > 2 else None)
    df["avg_dropout"] = df.drops.apply(lambda x: sum(x)/len(x))
    df["first_dropout"] = df.drops.apply(lambda x: x[0])
    df["last_dropout"] = df.drops.apply(lambda x: x[-1])
    df["second_dropout"] = df.drops.apply(lambda x: x[1] if len(x) > 2 else None)
    df["penultimate_dropout"] = df.drops.apply(lambda x: x[-2] if len(x) > 2 else None)
    df["third_dropout"] = df.drops.apply(lambda x: x[2] if len(x) > 4 else None)
    df["third_last_dropout"] = df.drops.apply(lambda x: x[-3] if len(x) > 4 else None)

    return df


def get_graph(minutes: int, strict: bool, trend: bool, mask: bool, x: str, dff: Optional[pd.DataFrame] = None, three_dim: Optional[str] = None) -> Figure:
    dff = manipulate(pd.read_sql("SELECT * FROM master_thesis.results_training", conn)) if dff is None else dff
    
    if x not in dff.columns:
        return None
    translation = {"epoch_ratio": "Epoch Ratio", "lr": "Learning Rate", "opt": "Optimizer", "n_layers": "Number of Layers", "first_layer": "first layer neurons", "second_layer": "second layer neurons", "third_layer": "third layer neurons", "avg_dropout": "average dropout ratio", "first_dropout": "dropout ratio first layer", "second_dropout": "dropout ratio second layer", "third_dropout": "dropout ratio third layer", "last_dropout": "dropout ratio last layer", "penultimate_dropout": "dropout ratio penultimate layer", "third_last_dropout": "dropout ratio third last layer"}
    msk = dff.dataset == "{'minutes': " + str(minutes) + ", 'strict': " + (str(strict) if minutes != 1 else 'False') + "}"
    dff = dff[msk] if mask else dff
    y = "f1_test" if x == "epoch_ratio" else "valid_f1"

    if x in ["opt", "n_layers"]:
        group_labels = [v for v in dff[x].unique()]
        hist_data = [[w for w in dff[dff[x] == v].valid_f1.values if str(w) not in ["None", "nan"]] for v in group_labels]

    # Create distplot with custom bin_size
    if three_dim and three_dim in dff.columns:
        fig = px.scatter_3d(dff, x=x, y=three_dim, z='valid_f1', color='dataset')
        fig.update_layout(scene=dict(yaxis_title=translation.get(three_dim), xaxis_title=translation.get(x), zaxis_title='F1 Score'))
    else:
        fig = ff.create_distplot(hist_data, group_labels, bin_size=.2) if x in ["opt", "n_layers"] else (px.scatter(dff, x=x, y=y, trendline='ols') if trend else px.scatter(dff, x=x, y=y, color='dataset'))
        fig.update_layout(yaxis_title=(translation.get(x) if x in ["opt", "n_layers"] else "F1 Score"), xaxis_title=(translation.get(x) if x not in ["opt", "n_layers"] else "F1 Score"))
    strict_or_not = ("strict minutes seperation" if strict else "no stricht minutes seperation")
    fig.update_layout(showlegend=(not mask), title=f"Distribution of {('validation' if  x != 'epoch_ratio' else 'test')} F1 Scores regarding the {translation.get(x)}{f' and {translation.get(three_dim)}' if three_dim else ''} variation<br>For the {('whole dataset' if not mask else f'dataset containing {minutes} minutes seperation with {strict_or_not}')}")
    fig.update_yaxes(visible=(x not in ["opt", "n_layers"]) or bool(three_dim))
    return fig


scipy.signal.signaltools._centered = _centered
