import optuna
from optuna.trial import TrialState
import torch
import torch.utils.data
from torch import nn, optim, utils

import copy
import numpy as np
import pandas as pd
import ray
import warnings
import json
import argparse

from sqlalchemy import create_engine
from tqdm import tqdm
from typing import Optional, Tuple
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.container import Sequential
from optuna.trial._trial import Trial

import logging

# define some basic things for the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
N_TRIALS = 100
MAX_LAYERS = 3
EPOCHS = 50

# define the minute and what dataset is loaded
MINUTES = 5

args = {}
if __name__ == "__main__":
    # setting up the flags for different settings
    parser = argparse.ArgumentParser(
        description="This module will optimize the hyperparameter for given dataset input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # define the minutes of the dataset
    parser.add_argument(
        "-m",
        "--minutes",
        type=int,
        help="set the minutes of the dataset. The value must between 2 and 9. Default value is 5",
    )
    # define the batch sizes
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        help="set the batches that will be loaded, default value is 32",
    )
    # define the trial numbers
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        help="set the number of trials, default value is 100",
    )
    # define the max layers
    parser.add_argument(
        "-l",
        "--maxlayer",
        type=int,
        help="set the number of max layers, default value is 3",
    )
    # define the max epochs
    parser.add_argument(
        "-e",
        "--maxepochs",
        type=int,
        help="set the number of max epochs, default value is 50 and the value can not be below 5",
    )
    # specify which dataset is used
    parser.add_argument(
        "-s",
        "--strict",
        action="store_true",
        help="set the strictness of the dataset. Default value if not selected is false. Strict means that if e.g. 3 Minutes is selected minutes 1.-3. and after that 4.-6. are selected. If not that would mean that 1.-3. is followed by 2.-4.",
    )

    args = parser.parse_args()
    args = vars(args)

strict = args.get("strict")
minutes = (
    min(9, max(1, args.get("minutes", MINUTES)))
    if args.get("minutes") is not None
    else MINUTES
)
batch_size = (
    args.get("batch", BATCH_SIZE) if args.get("batch") is not None else BATCH_SIZE
)
n_trials = (
    args.get("trials", N_TRIALS) if args.get("trials") is not None else N_TRIALS
)
max_layers = (
    args.get("maxlayers", MAX_LAYERS) if args.get("maxlayers") is not None else MAX_LAYERS
)
max_epochs = (
    args.get("maxepochs", EPOCHS) if args.get("maxepochs") is not None else EPOCHS
)
max_epochs = max(5, max_epochs)

# basic configurations for logging settings
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    filename=f"creation_{minutes}{('_2' if not strict else '')}.log",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# don't show warnings
warnings.filterwarnings("ignore")

with open("./config.json") as config_file:
    config = json.load(config_file)

ray.init(log_to_driver=False, num_cpus=6)

# create the engine
global conn
conn = create_engine(
    f'postgresql://{config.get("user")}:{config.get("password")}@{config.get("host")}:{config.get("port")}/{config.get("database")}'
).connect()


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


# @ray.remote
def gather_data(id: int) -> pd.DataFrame:
    """
    This function will gather information about the games

    Input:  id: The id of the game
    Output:     The DataFrame containg all the data
    """
    # query to select all information about the home and away data
    QUERY = f"""
        SELECT
            CASE WHEN match_id = '5' THEN 'anomaly' else 'normal' END AS result,
            content
        FROM
            master_thesis.data_group_{max(1, min(9, minutes))}{'_2' if (not strict) and (minutes != 1) else ''}
        WHERE
            match_id = '{id}'
        """

    # execute the query
    df = pd.read_sql(QUERY, conn)

    # put string in dict from content
    df["content"] = df["content"].apply(lambda x: x.replace("nan", "np.nan"))
    df["content"] = df["content"].apply(eval)

    # unpack the dictionary values
    content = pd.json_normalize(df.content)

    # concat the results
    return pd.concat([df[["result"]], content], axis=1)


def define_model(trial: Trial) -> Tuple[Sequential, int, list, list]:
    """
    Define the model based on some hyperparameters given by the trial from Optuna.
    The layers are created and the Dropout is for now only ReLu.
    Also the architecture is mirrored.
    Dropout is added as well from optuna optimization.

    Input:  trial:      The optuna trial which will get the different parameters
    Output: model:      The model for the given trial
            n_layers:   The numbers of layers in the model
            outs:       The output of the different hidden layers
            drops:      The dropout rates for the model
    """
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, max_layers)
    layers = []

    # define the input feature given by the dataset
    in_features = 872

    outs = []
    drops = []

    # loop over all the layers for the decoder
    for i in range(n_layers):
        # get the output features for this trial in this layer
        out_features = trial.suggest_int("n_units_l{}".format(i), 10, 700)
        # out_features = trial.suggest_int("n_units_l{}".format(i), 10, (outs[-1] if i != 0 else 700)))
        # add the layer
        layers.append(nn.Linear((in_features if i == 0 else outs[-1]), out_features))
        # put the output features in a list
        outs.append(out_features)
        # add the activation function
        layers.append(nn.ReLU())
        # find a dropout rate for this layer in this trial
        p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)
        # put the dropout rate in the list
        drops.append(p)
        # add the dropout layer
        layers.append(nn.Dropout(p))

    # for the encoder the mirrored architecture needs to be added
    for i in range(n_layers, 0, -1):
        # gather the given architecture from the decoder
        layers.append(nn.Linear(outs[i - 1], (outs[i - 2] if i != 1 else in_features)))
        # find the dropout ratio
        p = trial.suggest_float("dropout_l{}".format(i + n_layers), 0, 0.5)
        # add the ReLu activation layer, but not for the last layer
        if i != 1:
            layers.append(nn.ReLU())
        # put the dropout ratio in the list
        drops.append(p)
        # add the dropout layer
        layers.append(nn.Dropout(p))

    # return the model and the hyperparameter information
    return nn.Sequential(*layers), n_layers * 2, outs, drops


def get_datasets() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function returns the dataset split into train, test and anomaly.
    It will load the the data from the database and will transform it into a pytorch DataLoader

    Output: train_loader:   The DataLoader containing the training data
            test_loader:    The DataLoader containing the test data
            anomaly_loader: The DataLoader containing the anomaly game minutes
    """
    # load the the dataset from the database
    df = pd.concat([gather_data(id) for id in tqdm(get_ids())])

    # filter for anomalies
    anomaly_msk = df.result == "anomaly"
    anomaly = df[anomaly_msk]

    # rest will go into the train dataframe
    train = df[~anomaly_msk]
    # shuffle the training data
    train = train.sample(frac=1).reset_index(drop=True)

    # split in test and train datasets
    num_train = int(len(train) * 0.7)
    test = train[num_train:]
    train = train[num_train:]

    # load it into pytorch dataloader for the train set
    train = torch.tensor(train.drop("result", axis=1).values.astype(np.float32))
    train_tensor = utils.data.TensorDataset(train)
    train_loader = utils.data.DataLoader(
        dataset=train_tensor, batch_size=batch_size, shuffle=True
    )
    # load it into pytorch dataloader for the test set
    test = torch.tensor(test.drop("result", axis=1).values.astype(np.float32))
    test_tensor = utils.data.TensorDataset(test)
    test_loader = utils.data.DataLoader(
        dataset=test_tensor, batch_size=batch_size, shuffle=True
    )
    # load it into pytorch dataloader for the anomaly set
    anomaly = torch.tensor(anomaly.drop("result", axis=1).values.astype(np.float32))
    anomaly_tensor = utils.data.TensorDataset(anomaly)
    anomaly_loader = utils.data.DataLoader(
        dataset=anomaly_tensor, batch_size=batch_size, shuffle=True
    )

    return train_loader, test_loader, anomaly_loader


def get_f1_score(d: dict) -> float:
    """
    This function will return the F1 score based on the True positive, False negative and so on
    This metric will be used to identify the models that are performing well

    Input:  d:  The dictionary containing the amount of True positive, False negative and so on
    Output:     The F1 Score
    """
    # calculate the F1 Score and return it
    return d.get("True positive", 0) / (
        d.get("True positive", 0)
        + (d.get("False positive", 0) + d.get("False negative", 0)) / 2
    )


def objective(trial: Trial) -> float:
    """
    This function will train the model for the given trial
    During the trial an optimizer, the learning rate and the epochs will be selected
    The information during the training process will be logged to a logging file

    Input:  trial:      The current trial
    Output: f_score:    The F1 Score of this trial
    """
    # Generate the model.
    model, n_layers, outs, drops = define_model(trial)
    model = model.to(device)
    # mean-squared error loss
    criterion = nn.MSELoss()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # generate the learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # define/specify the model
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    # generate the epoch number
    epochs = trial.suggest_int("epochs", 5, max_epochs)

    # put the model info together for logging purposes
    model_info = str(
        {
            "lr": lr,
            "opt": optimizer_name,
            "n_layers": n_layers,
            "outs": outs,
            "drops": drops,
            "epochs": epochs,
            "minutes": minutes,
            "strict": strict,
        }
    )

    # Training of the model.
    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            # load it to the active device
            batch_features = torch.stack(batch_features).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            # loss_list += [criterion(y_pred[:,i,:], batch_features[:,i,:]) for i in range(y_pred.shape[-2])]

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # log the epoch training loss
        logging.info(
            "epoch : {}/{}, loss = {:.6f} for model ".format(epoch + 1, epochs, loss)
            + model_info
        )
        # put the model in evaluation mode
        eval_model = copy.deepcopy(model)
        eval_model.eval()

        # get the test loss
        test_loss = 0
        test_loss_list = []
        for batch_features in test_loader:
            batch_features = torch.stack(batch_features).to(device)
            # get the predictions/reproduction
            y_pred = eval_model(batch_features)
            # calculate the losses
            test_loss += criterion(y_pred, batch_features)
            test_loss_list += [
                criterion(y_pred[:, i, :], batch_features[:, i, :])
                for i in range(y_pred.shape[-2])
            ]

        # log the test loss
        logging.info(
            "test loss is {} for model ".format(test_loss / len(test_loader))
            + model_info
        )

        # get the anomaly loss
        anomaly_loss = 0
        anomaly_loss_list = []
        for batch_features in anomaly_loader:
            batch_features = torch.stack(batch_features).to(device)
            # get the prediction/reproduction
            y_pred = eval_model(batch_features)
            # calculate the losses
            anomaly_loss += criterion(y_pred, batch_features)
            anomaly_loss_list += [
                criterion(y_pred[:, i, :], batch_features[:, i, :])
                for i in range(y_pred.shape[-2])
            ]
        # log the anomaly loss
        logging.info(
            "anomaly loss is {} for model ".format(anomaly_loss / len(anomaly_loader))
            + model_info
        )
        # calculate the best F1 score
        f_score = max(
            [
                get_f1_score(
                    {
                        "False negative": i,
                        "False positive": int(sum([t >= v for t in test_loss_list])),
                        "True positive": len(anomaly_loss_list) - i,
                        "True negative": int(sum([t < v for t in test_loss_list])),
                        "threshold": v,
                    }
                )
                for i, v in enumerate(sorted(anomaly_loss_list))
            ]
        )
        # log the F1 Score
        logging.info("The F1 score is {} for model ".format(f_score) + model_info)

        # report the results
        trial.report(f_score, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return f_score


def main() -> None:
    # information about the dataset
    dataset = str({"minutes": minutes, "strict": strict})

    # initialize the finding for the best parameters
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # get further information about the the completed and pruned trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # log some further informations
    logging.info(f"Study statistics for the dataset {dataset}: ")
    logging.info("  Number of finished trials: " + str(len(study.trials)))
    logging.info("  Number of pruned trials: " + str(len(pruned_trials)))
    logging.info("  Number of complete trials: " + str(len(complete_trials)))

    # find the best trial and log these information
    logging.info("Best trial:")
    trial = study.best_trial

    logging.info("  Value: " + str(trial.value))

    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info("    {}: {}".format(key, value))


if __name__ == "__main__":
    # Get the dataset.
    train_loader, test_loader, anomaly_loader = get_datasets()
    main()
