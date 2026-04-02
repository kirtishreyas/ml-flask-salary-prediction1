import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# load dataset from csv file
df = pd.read_csv("dataset/salary_data.csv")

# take input column
X = df[["YearsExperience"]]

# take output column
y = df["Salary"]

# create linear regression model object
model = LinearRegression()

# train the model
model.fit(X, y)

# save trained model in model folder
joblib.dump(model, "model/model.pkl")

print("Model trained and saved successfully.")