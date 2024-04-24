from sqlalchemy import create_engine
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json
import numpy as np
import warnings
from collections import Counter
from plotly.graph_objs._figure import Figure
from typing import Optional, Union, List
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


def get_graph_test_f1(
    test: Optional[bool] = False,
    ba_data: Optional[bool] = False,
    minutes_restr: Optional[Union[bool, None, List[int]]] = False,
) -> Figure:
    """
    This function returns graphs regarding the general training result for the trained models
    Also in comparison to the bachelor thesis results
    Input:  test:           Optional boolean value indicating whether the test or validation results should be compared
            ba_data:        Optional boolean value to indicate whether the test data should be compared to the master thesis data
                            Only relevant if test parameter is True
            minutes_restr:  A optional list of containing only minutes that should be displayed
    """
    # load the data
    df = manipulate(pd.read_sql("SELECT * FROM master_thesis.results_training", conn))

    # look at only the final epoch
    df = df[df.epoch_ratio == 1]

    # gather the results for the best test result
    df_res = (
        df[(df.groupby(["minutes", "strict"]).f1_test.transform(max) == df["f1_test"])]
        .groupby(["minutes", "strict"])
        .agg(
            lambda x: {
                max(list(x.get("f1_test"))): dict(Counter(list(x.get("valid_f1"))))
            }
        )
        .reset_index(drop=False)
        .rename({0: "results"}, axis=1)
    )

    # write the test and valid F1 scores in seperate columns
    df_res["test_f1"] = df_res.results.apply(lambda x: list(x.keys())[0])
    df_res["valid_f1"] = df_res.results.apply(
        lambda x: {
            k: v / sum(list(list(x.values())[0].values()))
            for k, v in list(x.values())[0].items()
        }
    )

    # for representing the test results
    if test:
        # for comparing with the bachelor thesis results
        if ba_data:
            # gather the bachelor thesis results
            with open("../data/ba_f1.json") as f:
                ba_data = pd.DataFrame(
                    json.load(f).items(), columns=["model", "f1_score"]
                )

            # extract the minutes of the model
            ba_data["minutes"] = ba_data.model.apply(lambda x: x[x.find("__") + 2 :])
            ba_data["minutes"] = ba_data.minutes.apply(
                lambda x: int(x[: x.find("_")]) - 1
            )

            # get the best model performance for each dataset
            ba_data = (
                ba_data.groupby("minutes")
                .f1_score.max()
                .reset_index(drop=False)
                .rename({"f1_score": "test_f1"}, axis=1)
            )

            # rename the type to bachelor thesis
            ba_data["strict"] = "bachelor_thesis"

            # merge with other results
            df_res = pd.concat([df_res, ba_data])

        if minutes_restr:
            df_res = df_res[df_res.minutes.isin(minutes_restr)]

        # return the graph
        fig = px.bar(
            df_res.replace(True, "Dataset 1")
            .replace(False, "Dataset 2")
            .rename({"strict": "dataset"}, axis=1),
            x="minutes",
            y="test_f1",
            color="dataset",
            barmode="group",
        )

        # update the width of the bars
        fig.update_traces(width=0.25)
        fig.update_layout(
            title=f"F1 Test scores compared for different datasets",
        )

        return fig

    # for validation results gather the ratio for the various valid F1 scores and explode the dataframe
    df_res["valid_f1_ratio"] = df_res.valid_f1.apply(lambda x: list(x.values()))
    df_res["valid_f1"] = df_res.valid_f1.apply(lambda x: list(x.keys()))
    df_res = df_res.explode(["valid_f1", "valid_f1_ratio"])
    df_res["dataset"] = df_res.strict.replace(True, "Dataset 1").replace(
        False, "Dataset 2"
    )

    if minutes_restr:
        df_res = df_res[df_res.minutes.isin(minutes_restr)]

    # return the figure
    fig = px.bar(
        df_res,
        x="valid_f1",
        y="valid_f1_ratio",
        barmode="group",
        color="dataset",
        facet_row="minutes",
        category_orders={
            "minutes": [1, 2, 3, 4, 5, 6, 7, 8, 9]
            if not minutes_restr
            else minutes_restr
        },
        width=1200,
        height=(2400 if not minutes_restr else 2400 / (9 / len(minutes_restr))),
    )
    fig.update_layout(
        title=f"F1 Test scores compared for different datasets",
        xaxis=dict(tickmode="linear", tick0=0.5, dtick=1 / 6),
    )

    return fig


scipy.signal.signaltools._centered = _centered
