import itertools

import matplotlib
import xgboost as xgb
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import time

from dateutil.parser import parse
import pytz


# https://xgboost.readthedocs.io/en/latest/python/python_intro.html

MAX = 257622.0


def preprocess_data():

    # Adjustable parameters
    solar_file_name = "./data/Lakeside California/289KW_PV_System_Hourly.csv"
    weather_file_name = "./data/Lakeside California/LakesideCA_Solcast_15m.csv"

    # Build a temporary data to store data in
    delete_test_train_split()
    os.mkdir("./out/")

    # Load the solar data
    solar_data = pd.read_csv(solar_file_name)

    # Convert solar data into [datetime, generated]
    data = {}
    local_tz = pytz.timezone("America/Los_Angeles")
    for index, row in solar_data.iterrows():
        d, t, g = row["Date (dd/mm/yy)"], row["Time (12 Hour)"], row["Generated (W)"]

        date_time = parse(f"20{d[6]}{d[7]}-{d[3]}{d[4]}-{d[0]}{d[1]} {t}")
        date_time = local_tz.localize(date_time)
        date_time = date_time.astimezone(pytz.UTC)

        data[date_time] = g

    # Load the weather data
    weather_data = pd.read_csv(weather_file_name)
    weather_data = weather_data.iloc[434067:449082]

    # Pair the weather and solar data; place into new dataframe
    formatted_data = []
    for index, row in weather_data.iterrows():
        date_time = parse(row["PeriodStart"])
        date_time = date_time.astimezone(pytz.UTC)

        if date_time in data:
            formatted_data.append(
                {
                    "DateTime": date_time,
                    "Generated": data[date_time] / MAX,
                    "AirTemp": row["AirTemp"],
                    "Azimuth": row["Azimuth"],
                    "CloudOpacity": row["CloudOpacity"],
                    "DewpointTemp": row["DewpointTemp"],
                    "Dhi": row["Dhi"],
                    "Dni": row["Dni"],
                    "Ebh": row["Ebh"],
                    "Ghi": row["Ghi"],
                    "PrecipitableWater": row["PrecipitableWater"],
                    "RelativeHumidity": row["RelativeHumidity"],
                    "SnowDepth": row["SnowDepth"],
                    "SurfacePressure": row["SurfacePressure"],
                    "WindDirection10m": row["WindDirection10m"],
                    "WindSpeed10m": row["WindSpeed10m"],
                    "Zenith": row["Zenith"],
                }
            )

    data = pd.DataFrame(formatted_data)

    # Split the data into a test/train split
    data.dropna(how="any", inplace=True)
    data = data.sample(frac=1).reset_index(drop=True)
    length = len(data)
    split_length = int(length * 0.2)
    train, test = data[: (length - split_length)], data[(length - split_length) :]

    # Save the data
    train.to_csv(f"./out/train.csv", index=False)
    test.to_csv(f"./out/test.csv", index=False)


def delete_test_train_split():

    try:
        for file_name in os.listdir(f"{os.path.abspath('./out/')}"):
            os.unlink(f"{os.path.abspath('./out/')}/{file_name}")
        os.rmdir("./out/")
    except:
        return


def train_model():

    train = pd.read_csv("./out/train.csv")
    test = pd.read_csv("./out/test.csv")

    # Pull apart the independent and dependent variables
    train_target = train["Generated"]
    test_target = test["Generated"]
    train.drop(["DateTime", "Generated"], axis=1, inplace=True)
    test.drop(["DateTime", "Generated"], axis=1, inplace=True)

    # Build a XGB matrix for both train and test with their target variables
    train = xgb.DMatrix(train, train_target)
    test = xgb.DMatrix(test, test_target)

    # Build the XGB parameter list
    param = {}  # https://xgboost.readthedocs.io/en/latest/parameter.html
    param["booster"] = "gbtree"
    param["verbosity"] = 1  # 0 - silent, 1 - warn, 2 - info, 3 - debug
    param["nthread"] = 12  # defaults to max system threads when using CPU
    param["learning_rate"] = 0.005  # alias is eta
    param["max_depth"] = 6  # more complexity with higher depths
    param["subsample"] = 0.8  # cross-fold with 0.8 meaning 80% train, 20% valid
    param["lambda"] = 1.0  # L2 regularization term
    param["objective"] = "reg:logistic"
    param["eval_metric"] = ["mae", "auc"]

    # Train the model
    eval_list = [(test, "eval"), (train, "train")]
    num_round = 30_000
    start = time.time()
    bst = xgb.train(
        param,
        train,
        num_boost_round=num_round,
        evals=eval_list,
        early_stopping_rounds=20,
    )
    print(f"Duration: {time.time() - start}s")
    pickle.dump(bst, open("./out/bst_model.pck", "wb"))


def plot_model(bst):

    # Plot setup
    sns.set(font_scale=1.25, rc={"figure.figsize": (15, 10)})
    sns.set_style("ticks")  # ticks
    sns.set_palette("colorblind")

    # Plot the importance of each feature
    xgb.plot_importance(bst)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig("./out/importance.png", bbox_inches="tight")
    clear_plots()

    # Plot setup
    sns.set(font_scale=1.25, rc={"figure.figsize": (20, 10)})
    sns.set_style("whitegrid")  # ticks
    sns.set_palette("colorblind")

    # Plot a decision tree
    matplotlib.rcParams["figure.dpi"] = 1080
    xgb.plot_tree(bst, num_trees=4, rankdir="LR")
    plt.savefig("./out/tree.png")
    clear_plots()


def test_model():

    # How much of a moving average to have
    smoothing_step = 1500

    # Load data
    test = pd.read_csv("./out/test.csv")

    # Store dates for plotting output
    start_date = test["DateTime"][0]
    end_date = test["DateTime"][len(test) - smoothing_step]
    dates = test["DateTime"]

    # Format data
    test_target = test["Generated"]
    test.drop(["DateTime", "Generated"], axis=1, inplace=True)
    test = xgb.DMatrix(test)

    # Load model and plot it
    bst = pickle.load(open("./out/bst_model.pck", "rb"))
    plot_model(bst)

    # Create target vs prediction
    pred = bst.predict(test)
    actual = test_target

    # Scale data from 0-1 to 0-MAX
    actual *= MAX
    pred *= MAX

    # Create a smoothing average
    smooth_pred = []
    smooth_actual = []
    error = 0
    for i in range(len(pred) - smoothing_step):
        smooth_pred.append(np.mean(pred[i : i + smoothing_step]))
        smooth_actual.append(np.mean(actual[i : i + smoothing_step]))
        error += abs(pred[i] - actual[i])
    error /= len(smooth_pred)

    # Line plot setup
    sns.set(font_scale=1.25, rc={"figure.figsize": (20, 10)})
    sns.set_style("whitegrid")  # ticks
    sns.set_palette("colorblind")

    # Decrease tick quantity
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))

    # Plot lines
    sns.lineplot(
        x=dates[: len(dates) - smoothing_step], y=smooth_pred, label="Prediction"
    )
    sns.lineplot(
        x=dates[: len(dates) - smoothing_step], y=smooth_actual, label="Actual"
    )
    plt.xlabel(f"Date")
    plt.ylabel(f"Energy Generation (Watts)")
    plt.title(
        f"Average Energy Generation Per Day From {end_date} to {start_date}\nAbsolute Error: {error:.2f} Watts"
    )
    plt.savefig("./out/error.png")
    clear_plots()

    # Scatter plot setup
    sns.set(font_scale=1.25, rc={"figure.figsize": (10, 10)})
    sns.set_style("whitegrid")  # ticks
    sns.set_palette("colorblind")

    sns.scatterplot(x=actual, y=pred)
    plt.title("Solar Energy Generation In 15 Minute Intervals")
    plt.xlabel("Actual Generation In Watts")
    plt.ylabel("Predicted Generation In Watts")
    plt.xlim(0.0, MAX)
    plt.ylim(0.0, MAX)
    plt.savefig("./out/scatterplot.png")
    clear_plots()


def clear_plots():
    plt.clf()
    import importlib

    importlib.reload(matplotlib)
    importlib.reload(plt)
    importlib.reload(sns)


if __name__ == "__main__":
    preprocess_data()
    train_model()
    test_model()

