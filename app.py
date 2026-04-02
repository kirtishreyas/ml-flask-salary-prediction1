from flask import Flask, render_template, request
import joblib
import pandas as pd
from config.config import HOST, PORT, DEBUG
from utils.helper import convert_to_float

# create flask app
app = Flask(__name__)

# load trained model
model = joblib.load("model/model.pkl")


# home page
@app.route('/')
def home():
    return render_template("home.html")


# prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get input from html form
        years_exp = request.form['years_experience']

        # convert text input to float
        years_exp = convert_to_float(years_exp)

        # create dataframe for prediction
        input_df = pd.DataFrame([[years_exp]], columns=["YearsExperience"])

        # predict salary
        prediction = model.predict(input_df)[0]

        # round value
        prediction = round(prediction, 2)

        # show result page
        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return f"Error occurred: {e}"


# run flask app
if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)