from flask import Flask,render_template,request,jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/vehicle_model.pkl','rb'))
unique_values = pickle.load(open('model/unique_values.pkl','rb'))

@app.route('/')
def base():
    return render_template('index.html',unique_values=unique_values)


@app.route('/predict',methods=['POST'])
def predict():
     # Get data from the form
    data = request.form.to_dict()

    # Convert data to a DataFrame
    input_data = pd.DataFrame([data])
    
    # Ensure input_data has the same structure as the training data
    input_data = input_data[[
        'make', 'model', 'type', 'engine', 'cylinders', 'fuel', 'mileage', 
        'transmission', 'trim', 'body', 'doors', 'exterior_color', 'interior_color', 'drivetrain','year'
    ]]

    # Preprocess and predict
    input_data_preprocessed = model.named_steps['preprocess'].transform(input_data)
    prediction_log = model.named_steps['regressionmodel'].predict(input_data_preprocessed)
    print(prediction_log)
    return render_template('index.html',prediction_log=prediction_log,unique_values=unique_values)


if __name__ == '__main__':
    app.run(debug=True)