import streamlit as st
import pickle
import numpy as np
import pandas as pd


def main():

	with open('best_model.pkl', "rb") as file:
		model = pickle.load(file)


	def user_input_predictors():
		st.sidebar.header('Please enter the values:')

		sidebar = []
		for i in range(1,33):
			SensVal = st.sidebar.number_input(f"Sensor {i}")
			sidebar.append(SensVal)

		return np.array(sidebar).reshape(1, -1)

	st.title("Target Score Prediction")
	st.write("This dashboard can be used to predict the target score value of a paper machine based on the input features")

	input_predictors = user_input_predictors()
	st.subheader("User Input")
	st.write(pd.DataFrame(input_predictors, columns=[f'sensor_{i}' for i in range(1,33)]))

	if st.button('Predict Target'):
		prediction = model.predict(input_predictors)
		st.success(f'Predicted target score is {prediction[0]}')
		#st.write(f'{prediction}')


if __name__ == '__main__':
	main()