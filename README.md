# SolarEnergyPrediction

Use historical energy production values along with weather predictions to forecast photovoltaic energy production.

## To run this code, you must have installed

### Machine learning packages

* xgboost          (pip install xgboost)

### Plotting packages

* matplotlib       (pip install matplotlib)
* seaborn          (pip install seaborn)

### General computing packages

* pandas           (pip install pandas)
* numpy            (pip install numpy)
* pytz             (pip install pytz)

### Webscraping packages

* beautiful soup 4 (pip install beautifulsoup4)
* selenium         (pip install selenium)

### Compiled pip install command

The full command is `pip install xgboost beautifulsoup4 selenium pandas numpy matplotlib seaborn pytz`.

## Running this code

The `model.py` file contains the creation, testing, and plotting of the xgboosted model.
The `scrape_website.py` file is for collecting the solar energy output data.

For an easy to run script, run the `src/model.py` file while in the project directory.
