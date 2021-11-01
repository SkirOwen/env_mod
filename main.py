import os
import sys

import numpy as np

import sklearn
import pandas as pd
import statsmodels as sm

import matplotlib.pyplot
import seaborn as sn


def poly_fit(n: int = 3):


	for i in range(1, n+1):
		print("fitting poly degree", i)




def main():
	filename = ""
	df = pd.read_csv(os.path.join("./dataset", filename))

	y = df.y
	x = df.x

	n = 3

	poly_fit(n)

if __name__ == '__main__':
	main()
