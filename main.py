#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot
import seaborn as sn

# A good place to start to see the conversion between Python and R
# https://towardsdatascience.com/cheat-sheet-for-python-dataframe-r-dataframe-syntax-conversions-450f656b44ca


def predict_interval(model, x, interval: str="confidence", level: int= 0.95):
	"""fit, lb, ub
	function(object, newdata, se.fit = FALSE, scale = NULL, df = Inf,
		interval = c("none", "confidence", "prediction"),
		level = .95,  type = c("response", "terms"),
		terms = NULL, na.action = na.pass, pred.var = res.var/weights,
		weights = 1, ...)
	"""
	pass


def poly_fit(x, y, n: int = 3):
	""""""
	# bic = [None for i in range(n)]
	bic_v = np.zeros(n)
	aic_v = np.zeros(n)
	aicc_v = np.zeros(n)

	for i in range(1, n+1):
		print("fitting poly degree", i)
		poly_reg = PolynomialFeatures(degree=i)
		x_poly = poly_reg.fit_transform(x)

		model = sm.OLS(y, x_poly).fit()
		y_pred = model.predict(x_poly)
		# pol_reg = LinearRegression(x)
		# pol_reg.fit(X_poly, y)

		plt.scatter(x, y)
		plt.plot(x, y_pred)
		plt.title(f"Degree {i}")
		plt.show()

	return

	#
	# 	# fig.savefig(os.path.join("./plots", f"degree{i}.png"))
	# 	plt.show()
	#
	#
	# 	# Predict 95% confidence model error bounds
	# 	t = pd.DataFrame(x, columns=["X"], index=range(1, len(x) + 1))
	# 	# this is to have exactly the same behaviour as data.frame(x) in R
	#
	# 	# Add model error bounds
	#
	# 	bic_v[i] = 0
	# 	aic_v[i] = 0
	# 	aicc_v[i] = 0
	#
	# return bic_v, aic_v, aicc_v


def main():
	filename = "MyData_Practical.csv"
	df = pd.read_csv(os.path.join("./dataset", filename))

	x = df.iloc[:, 1:2].values
	y = df.iloc[:, 2].values

	# .iloc uses array index, it return sa column vectors
	# doing this for the fitting later
	# df.x or df.V1 could work most of the time, but output a line vector
	# you may wish to look into .loc as it uses the column name

	n = 4

	poly_fit(x, y, n)
	# bic_v, aic_v, aicc_v = poly_fit(x, y, n)


if __name__ == '__main__':
	main()
