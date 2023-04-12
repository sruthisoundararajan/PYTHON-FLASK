from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and scaler objects
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the date and time input from the form
    date = pd.to_datetime(request.form['date'])
    time = pd.to_datetime(request.form['time']).time()
    
    # Combine the date and time into a single datetime object
    datetime = pd.to_datetime(str(date.date()) + ' ' + str(time))
    
    # Convert the datetime object to a timestamp
    timestamp = pd.Timestamp(datetime)
    
    # Create a DataFrame with the timestamp as the index
    data = pd.DataFrame(index=[timestamp])
    
    # Scale the input data
    data_scaled = scaler.transform(data)
    
    # Make the prediction
    prediction = model.predict(data_scaled)[0]
    
    # Render the result template with the predicted value
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
