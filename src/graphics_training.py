from sqlalchemy import create_engine
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json
import numpy as np
import warnings
from plotly.graph_objs._figure import Figure
from typing import Optional
import scipy.signal.signaltools

# don't show warnings
warnings.filterwarnings("ignore")

with open("../config.json") as config_file:
    config = json.load(config_file)

# create the engine
conn = create_engine(
    f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
).connect()

# This function is needed for the trendline in the plot
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
    # since the number of layers is not a continuum it will be displayed as a string
    df["n_layers"] = df.n_layers.astype(str)

    # gather information regarding model and dataset
    df["model"] = df.apply(
        lambda x: str(
            {
                "lr": x["lr"],
                "opt": x["opt"],
                "n_layers": x["n_layers"],
                "outs": x["outs"],
                "drops": x["drops"],
                "epochs": x["epochs"],
            }
        ),
        axis=1,
    )
    df["dataset"] = df.apply(
        lambda x: str({"minutes": x["minutes"], "strict": x["strict"]}), axis=1
    )

    # calculate the epoch ratio
    df["epoch_ratio"] = df.epoch / df.epochs

    # drops and outs should be in list form not in string form
    for c in ["drops", "outs"]:
        df[c] = df[c].apply(eval)

    # extract neuron number of the list
    df["first_layer"] = df.outs.apply(lambda x: x[0])
    df["second_layer"] = df.outs.apply(lambda x: x[1] if len(x) > 1 else None)
    df["third_layer"] = df.outs.apply(lambda x: x[2] if len(x) > 2 else None)

    # extract information regarding the dropout ratio
    df["avg_dropout"] = df.drops.apply(lambda x: sum(x) / len(x))
    df["first_dropout"] = df.drops.apply(lambda x: x[0])
    df["last_dropout"] = df.drops.apply(lambda x: x[-1])
    df["second_dropout"] = df.drops.apply(lambda x: x[1] if len(x) > 2 else None)
    df["penultimate_dropout"] = df.drops.apply(lambda x: x[-2] if len(x) > 2 else None)
    df["third_dropout"] = df.drops.apply(lambda x: x[2] if len(x) > 4 else None)
    df["third_last_dropout"] = df.drops.apply(lambda x: x[-3] if len(x) > 4 else None)

    return df


def get_graph(
    minutes: int,
    strict: bool,
    trend: bool,
    mask: bool,
    x: str,
    dff: Optional[pd.DataFrame] = None,
    three_dim: Optional[str] = None,
) -> Figure:
    """
    This function will return a Figure of the training results
    It will be graph displayed showing the F1 results under different variations

    Input:  minutes:    The number of minutes in the dataset
            strict:     Boolean value regarding the dataset type
            trend:      Boolean value regarding the display of a trend line
            mask:       Boolean value whether the whole dataset should be investigated or only specific part
            x:          The x axis in the graph
            dff:        Optional given DataFrame w/ the trainings resulst, if not given it will be loaded
            three_dim:  Optional second column under which the F1 score variance will be investigated
    """
    # either load the DataFrame or take the given one
    dff = (
        manipulate(pd.read_sql("SELECT * FROM master_thesis.results_training", conn))
        if dff is None
        else dff
    )

    # This dict will be needed later for purposes regarding the translation of the cryptic columns
    translation = {
        "epoch_ratio": "Epoch Ratio",
        "lr": "Learning Rate",
        "opt": "Optimizer",
        "n_layers": "Number of Layers",
        "first_layer": "first layer neurons",
        "second_layer": "second layer neurons",
        "third_layer": "third layer neurons",
        "avg_dropout": "average dropout ratio",
        "first_dropout": "dropout ratio first layer",
        "second_dropout": "dropout ratio second layer",
        "third_dropout": "dropout ratio third layer",
        "last_dropout": "dropout ratio last layer",
        "penultimate_dropout": "dropout ratio penultimate layer",
        "third_last_dropout": "dropout ratio third last layer",
    }

    # if the x axis is not in the columns of the DataFrame None will be returned
    if x not in translation.keys():
        return None
    
    # write down the mask
    msk = (
        dff.dataset
        == "{'minutes': "
        + str(minutes)
        + ", 'strict': "
        + (str(strict) if minutes != 1 else "False")
        + "}"
    )
    # and apply it if the mask value is given
    dff = dff[msk] if mask else dff

    # define the y value (valid or test) F1
    y = "f1_test" if x == "epoch_ratio" else "valid_f1"

    # for the not continous variables a different figure will be created
    if x in ["opt", "n_layers"] and (not three_dim):
        group_labels = [v for v in dff[x].unique()]
        hist_data = [
            [
                w
                for w in dff[dff[x] == v].valid_f1.values
                if str(w) not in ["None", "nan"]
            ]
            for v in group_labels
        ]

    # create the plots based on the given data and name the axis
    if three_dim and three_dim in dff.columns:
        fig = px.scatter_3d(dff, x=x, y=three_dim, z="valid_f1", color="dataset")
        fig.update_layout(
            scene=dict(
                yaxis_title=translation.get(three_dim),
                xaxis_title=translation.get(x),
                zaxis_title="F1 Score",
            )
        )
    else:
        fig = (
            ff.create_distplot(hist_data, group_labels, bin_size=0.2)
            if x in ["opt", "n_layers"]
            else (
                px.scatter(dff, x=x, y=y, trendline="ols")
                if trend
                else px.scatter(dff, x=x, y=y, color="dataset")
            )
        )
        fig.update_layout(
            yaxis_title=(
                translation.get(x) if x in ["opt", "n_layers"] else "F1 Score"
            ),
            xaxis_title=(
                translation.get(x) if x not in ["opt", "n_layers"] else "F1 Score"
            ),
        )
    
    # give the figure a title
    strict_or_not = (
        "strict minutes seperation" if strict else "no stricht minutes seperation"
    )
    fig.update_layout(
        showlegend=(not mask),
        title=f"Distribution of {('validation' if  x != 'epoch_ratio' else 'test')} F1 Scores regarding the {translation.get(x)}{f' and {translation.get(three_dim)}' if three_dim else ''} variation<br>For the {('whole dataset' if not mask else f'dataset containing {minutes} minutes seperation with {strict_or_not}')}",
    )
    # display the y axis label only if it makes sense
    fig.update_yaxes(visible=(x not in ["opt", "n_layers"]) or bool(three_dim))
    return fig


scipy.signal.signaltools._centered = _centered
