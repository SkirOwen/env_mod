#!/usr/bin/env python3

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.eval_measures import aic, bic, aicc

import seaborn as sns

sns.set_theme()
# this is to make the matplotlib looks the same as the seaborn one

# A good place to start to see the conversion between Python and R
# https://towardsdatascience.com/cheat-sheet-for-python-dataframe-r-dataframe-syntax-conversions-450f656b44ca
# https://ostwalprasad.github.io/machine-learning/Polynomial-Regression-using-statsmodel.html

# DISCLAIMER: plotting things is really easy with the library `seaborn`, but doesn't provide a lot of information
# about the error of the fitting.
# The alternative is to do it manually, using sklearn and statsmodels
# There is a weird quirk about the aic and bic, they are not the same as with R. It seems to come from the way R
# counts the parameters of the model.

# NOTE: in the function declaration, I sometimes indicate the type of the variables (like: x_name: str)
# Python does NOT care about it, it simply for the user (and me) to know quickly what I can put it or not.
# It does not perform any checks, but it can help you when you run into an error.
# You can also indicates what the function return, with ->, like:
# def foo(bar) -> int:


def sns_predict(x_name: str, y_name: str, data: DataFrame, degree: float, ci: int=95) -> None:
	"""
	Implementation of the plotting with CI with seaborn.
	It only does the plot (with the fitting under the hood)

	Parameters
	----------
	x_name : str
		Name of the x column in the pd.DataFrame
	y_name : str
		Name of the y column in the pd.DataFrame
	data : DataFrame
		Pandas DataFrame
	degree : float
		degree of the fit, float numbers work, even negative :)
	ci : int in [0, 100], optional
		Confidence interval
	"""
	# sns.scatterplot(x=df.V1.values, y=df.V2.values)
	sns.regplot(x=x_name, y=y_name, data=data, order=degree, ci=ci)
	plt.savefig(os.path.join("./plots", f"seaborn.png"), dpi=400)
	plt.show()


def poly_fit(
				x_name: str,
				y_name: str,
				df: DataFrame,
				n: int = 3,
				ci_level=0.05,
				R_values: bool = True
			) -> Tuple[np.array, np.array, np.array]:
	"""
	Function to fit a polynomial regression for different degree and measure the AIC, BIC, AICc.
	Plot the fitting with CI, and saves it to `./plots`

	Parameters
	----------
	x_name : str
		Name of the x column in the pd.DataFrame
	y_name : str
		Name of the y column in the pd.DataFrame
	df : DataFrame
		Pandas DataFrame
	n : int, optional
		max degree of the polynomial fit, default is 3
	ci_level : float in [0, 1], optional
		confidence interval, 0.05 correspond to a CI at 95%, default is 0.05
	R_values : bool, optional
		get the same values for AIC, BIC, AICc as in R, default is True

	Returns
	-------
	aic_v : np.array
		numpy array of (1, n) with the values of the AIC for the n degrees
	bic_v : np.array
		numpy array of (1, n) with the values of the BIC for the n degrees
	aicc_v : np.array
		numpy array of (1, n) with the values of the AICc for the n degrees

	"""

	x = df.iloc[:, 1:2].values
	y = df.iloc[:, 2].values
	# .iloc uses array index, it return sa column vectors
	# doing this for the fitting later
	# df.x or df.V1 could work most of the time, but output a line vector
	# you may wish to look into .loc as it uses the column name

	bic_v = np.zeros(n)
	aic_v = np.zeros(n)
	aicc_v = np.zeros(n)
	# Creating array of zeros of size n using numpy, because it is faster then the python list for later
	# The pythonic alternative would be:
	# bic = [None for i in range(n)]

	for i in range(1, n+1):
		# python loop are up to n but doesn't include n, also by defaults starts at 0
		# and list indexes start at 0
		print("fitting poly degree", i)
		poly_reg = PolynomialFeatures(degree=i)
		x_poly = poly_reg.fit_transform(x)

		x_fit = sm.add_constant(x_poly)

		model = sm.OLS(y, x_fit).fit()

		y_pred = model.get_prediction(x_fit).summary_frame(ci_level)

		plt.plot(df[x_name], y_pred["mean_ci_lower"], linestyle="--", color="orange")
		plt.plot(df[x_name], y_pred["mean_ci_upper"], linestyle="--", color="orange")
		plt.fill_between(df[x_name], y_pred["mean_ci_lower"], y_pred["mean_ci_upper"], alpha=.25,
							label=f"CI ({(1-ci_level) * 100}%)")

		plt.scatter(df[x_name], df[y_name], label="data", linewidth=1.5, facecolors="none", edgecolors="b")

		plt.plot(df[x_name], y_pred["mean"], label="Regression Line")

		plt.legend()

		plt.xlabel(x_name)
		plt.ylabel(y_name)
		plt.title(f"Degree {i}")

		plt.savefig(os.path.join("./plots", f"degree{i}.png"), dpi=400)
		plt.show()

		# computing the values by hand to make the same computation as in R, see next comment
		llf = model.llf
		nobs = len(model.fittedvalues)
		df_modelwc = len(model.params) + 1      # need to add one to have the same result as R

		if R_values:
			aic_v[i-1] = aic(llf, nobs, df_modelwc)
			bic_v[i-1] = bic(llf, nobs, df_modelwc)
			aicc_v[i-1] = aicc(llf, nobs, df_modelwc)
		else:
			aic_v[i-1] = model.aic
			bic_v[i-1] = model.bic
			aicc_v[i-1] = aicc(llf, nobs, df_modelwc-1)

	return aic_v, bic_v, aicc_v


def plot_fit_criteria(name: str, values) -> None:
	"""
	Small function to plot the AIC, BIC, AICc

	Parameters
	----------
	name : str,
		name of the criterion to plot, for titles and axes
	values : array,
		array of the values to plot
	"""
	x = [i for i in range(1, len(values) + 1)]

	plt.scatter(x, values, linewidth=1.5, facecolors="none", edgecolors="b")
	plt.plot(x, values)

	plt.xlabel("Order")
	plt.ylabel(name)
	plt.title(name)

	plt.savefig(os.path.join("./plots", f"{name}.png"), dpi=400)
	plt.show()


def main():
	# It is better to put a main() function instead of in the `if __name__ == "__main__"`
	# the main make all the variable as local variable, whereas variable in the if __name__ make them global
	# and can therefore be imported when using `import`
	# see:
	filename = "MyData_Practical.csv"
	df = pd.read_csv(os.path.join("./dataset", filename))

	x_name, y_name = "V1", "V2"

	n = 7

	aic_v, bic_v, aicc_v = poly_fit(x_name, y_name, df, n)

	plot_fit_criteria("AIC", aic_v)
	plot_fit_criteria("BIC", bic_v)
	plot_fit_criteria("AICc", aicc_v)


if __name__ == '__main__':
	# this means only execute the code if the file is executed as a script (not an import)
	# to understand this look: https://stackoverflow.com/questions/419163/what-does-if-name-main-do
	main()
