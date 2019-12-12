# SolarEnergyPrediction

Use historical energy production values along with weather predictions to forecast photovoltaic energy production.

## To run this code, you must have installed

* xgboost          (pip install xgboost)
* beautiful soup 4 (pip install beautifulsoup4)
* selenium         (pip install selenium)
* pandas           (pip install pandas)
* numpy            (pip install numpy)
* netCDF           (pip install netCDF4)
* matplotlib       (pip install matplotlib)
* seaborn          (pip install seaborn)
* pytz             (pip install pytz)

The full command is `pip install xgboost beautifulsoup4 selenium pandas numpy netCDF4 matplotlib seaborn pytz`.

## Running this code

The `model.py` file contains the creation, testing, and plotting of the xgboosted model.
The `scrape_website.py` file is for collecting the solar energy output data.

For an easy to run script, run the `run_me.py` file.
