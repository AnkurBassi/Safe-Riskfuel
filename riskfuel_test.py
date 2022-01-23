import pandas as pd
import argparse
import sys

# Load whatever imports you need, but make sure to add them to the requirements.txt file.

# My imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.tabular.all import *



def riskfuel_test(df: pd.DataFrame) -> float:
    """
    Riskfuel Testing Function
    by <team-name>: <member_1> <member_2> .... <member_k>

    arguments: pandas DataFrame type with the following columns.. ['S','K','T','r','sigma','value'] all are of type float32
    ouputs: mean absolute error (float)

    Once you have finished model training/developemnt you must save the model within the repo and load it in using this function.

    You are free to import any python packages you desire but you must add them to the requirements.txt file.

    This function must do the following:
        - Successfully load your own model.
        - Take in a dataframe consisting of (N x 6) float32's.
        - Take the (N x 5) columns regarding the inputs to the pricer ['S','K','T','r','sigma'] and have your model price them.
        - Return the Mean  Absolute Error of the model.

    Do not put the analytic pricer as part of your network.
    Do not do any trickery with column switching as part of your answer.

    These will be checked by hand, any gaslighting will result in automatic disqualification.

    The following example has been made available to you.
    """

    # TEAM DEFINITIONS.
    team_name = "Safe Riskfuel"  # adjust this
    members = ["Ankur Bassi"]  # adjust these

    print(f"\n\n ============ Evaluating Team: {team_name} ========================= ")
    print(" Members :")
    for member in members:
        print(f" {member}")
    print(" ================================================================ \n")

    # ===============   Example Code  ===============

    # My model uses PyTorch but you can use whatever package you like,
    # as long you write code to load it and effectively calculate the mean absolute aggregate error.

    # # LOAD MODEL
    learn = load_learner("file", cpu=True)

    x = df[["S", "K", "T", "r", "sigma"]]
    y = df[["value"]]
    x=x-x.mean()
    x=x/(x.max()-x.min())
    dl = learn.dls.test_dl(x)
    # # Pass data through model
    y_hat = learn.get_preds(dl=dl)[0]#my code
    #
    # # Calculate mean squared error
    result = F.mse_loss(torch.tensor(np.array(y_hat)), torch.tensor(np.array(y)))

    # Return performance metric; must be of type float
    return result.item()




def get_parser():
    """Parses the command line for the dataframe file name"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_frame_name", type=str)
    return parser


def main(args):
    """Parses arguments and evaluates model performance"""

    # Parse arguments.
    parser = get_parser()
    args = parser.parse_args(args)

    # Load DataFrame and pass through riskfuel_test function.
    df = pd.read_csv(args.data_frame_name)

    performance_metric = riskfuel_test(df)

    # Must pass this assertion
    assert isinstance(performance_metric, float)

    print(f" MODEL PERFORMANCE: {performance_metric} \n\n")


if __name__ == "__main__":
    main(sys.argv[1:])
