# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import math
import matplotlib.pyplot as plt
import numpy as np
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    fits = []
    for deg in degs:
        fits.append(np.polyfit(x, y, deg))
    return fits


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    ss_res = np.sum((y - estimated)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)


def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        # use np.polyval to calculate the estimated values
        # then calculate the r2_score for each model
        estimated = np.polyval(model, x)
        r2_score = r_squared(y, estimated)
        degree = len(model) - 1
        # plot the model
        plt.plot(x, y, 'bo', label='Measured Points')
        plt.plot(x, estimated, 'r', label=f'Model of degree {degree}')
        plt.xlabel('Years')
        plt.ylabel('Degrees in Celsius')
        plt.title(f'Model of degree {degree}\nR2 = {r2_score:.3f}')
        plt.legend(loc='best')
        # if the model has 2 coefficients then it's linear
        # calculate se over slope
        if degree == 1:
            slope_se = se_over_slope(x, y, estimated, model)
            plt.title(f'Linear Model\nR2 = {r2_score:.3f}\nStandard error-to-slope ratio = {slope_se:.3f}')
        plt.show()


def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    national_temp_by_year = []
    for year in years:
        # first average each year's temperatures (daily) for each city
        # then average the cities averages
        national_temp = np.mean(
            [np.mean(climate.get_yearly_temp(city, year)) for city \
             in multi_cities])
        # append to the average by year
        national_temp_by_year.append(national_temp)
    return national_temp_by_year


def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    # list comprehension of the average last window length items in the array
    # using max(0, i-window-length+1) to only average over the number of items
    # from the begining of the list
    y_moving_average = [np.mean(y[max(0, i-window_length+1): i+1]) for i in \
                        range(len(y))]
    return y_moving_average


def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    rmse = math.sqrt(sum((y - estimated) ** 2) / len(y))
    return rmse


def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    national_stds = []
    for year in years:
        national_std = np.std(
            np.mean([climate.get_yearly_temp(city, year) \
                     for city in multi_cities], axis=0)
        )
        national_stds.append(national_std)
    return national_stds


def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        # use np.polyval to calculate the estimated values
        # then calculate the r2_score for each model
        estimated = np.polyval(model, x)
        # using RMSE instead of R2
        rmse_score = rmse(y, estimated)
        degree = len(model) - 1
        # plot the model
        plt.plot(x, y, 'bo', label='Measured Points')
        plt.plot(x, estimated, 'r', label=f'Model of degree {degree}')
        plt.xlabel('Years')
        plt.ylabel('Degrees in Celsius')
        plt.title(f'Model of degree {degree}\nRMSE = {rmse_score:.3f}')
        plt.legend(loc='best')
        plt.show()

if __name__ == '__main__':

    pass 

    # Part A.4
    # define the climate class by loading the file
    # climate = Climate('src/ps5/data.csv')
    # get the years, month, day and city for the data sample
    # years = np.array(TRAINING_INTERVAL)
    # month = 1
    # day = 10
    # city = 'NEW YORK'
    # # get the temperatures for the date from the range of years
    # temps = np.array(
    #     [climate.get_daily_temp(city, month, day, year) for year in years])
    # temps = np.array(
    #     [np.mean(climate.get_yearly_temp(city, year)) for year in years])
    # # fit the linear model
    # model = generate_models(years, temps, [1])
    # evaluate_models_on_training(years, temps, model)

    # Part B
    # climate = Climate('src/ps5/data.csv')
    # years = np.array(TRAINING_INTERVAL)
    # temps = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
    # model = generate_models(years, temps, [1])
    # evaluate_models_on_training(years, temps, model)

    # Part C
    # climate = Climate('src/ps5/data.csv')
    # years = np.array(TRAINING_INTERVAL)
    # temps = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
    # temps_moving_avg = moving_average(temps, window_length=5)
    # model = generate_models(years, temps_moving_avg, [1])
    # evaluate_models_on_training(years, temps_moving_avg, model)

    # Part D.2
    # climate = Climate('src/ps5/data.csv')
    # years_training = np.array(TRAINING_INTERVAL)
    # temps_training = moving_average(
    #     gen_cities_avg(climate, CITIES, TRAINING_INTERVAL),
    #     5)
    # model = generate_models(years_training, temps_training, [1])
    # years_test = np.array(TESTING_INTERVAL)
    # temps_test = moving_average(
    #     gen_cities_avg(climate, CITIES, TESTING_INTERVAL),
    #     5
    # )
    # evaluate_models_on_testing(years_test, temps_test, model)

    # Part I
    # climate = Climate('src/ps5/data.csv')
    # years_training = np.array(TRAINING_INTERVAL)
    # temps_training = moving_average(
    #     gen_cities_avg(climate, CITIES, TRAINING_INTERVAL),
    #     5)
    # models = generate_models(years_training, temps_training, [1, 2, 20])
    # evaluate_models_on_training(years_training, temps_training, models)
    # years_test = np.array(TESTING_INTERVAL)
    # temps_test = moving_average(
    #     gen_cities_avg(climate, CITIES, TESTING_INTERVAL),
    #     5
    # )
    # evaluate_models_on_testing(years_test, temps_test, models)

    # Part E
    climate = Climate('src/ps5/data.csv')
    years_training = np.array(TRAINING_INTERVAL)
    temps_training = moving_average(
        gen_std_devs(climate, CITIES, TRAINING_INTERVAL),
        5)
    models = generate_models(years_training, temps_training, [1])
    evaluate_models_on_training(years_training, temps_training, models)