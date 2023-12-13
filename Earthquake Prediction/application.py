from flask import Flask, render_template, request
from flask import Flask, jsonify
import folium
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pickle
import plotly.express as px
import numpy as np
import requests
import jsonify

app = Flask(__name__)

base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
df = pd.read_csv("earthquaketill2023.csv")
data = []
params = {
            "format": "geojson",
            "starttime": 2023-10-12,
            "endtime": datetime.now(),
            "minmagnitude": 5.0,
            "limit": 20000,  # Adjust limit based on your needs
        }

response = requests.get(base_url, params=params)

if response.status_code == 200:
    earthquake_data = response.json()
    features = earthquake_data.get("features", [])
    data.extend(features)
else:
    print("Error")

columns = ["Date", "Magnitude", "Latitude", "Longitude", "Depth", "Location"]
x = pd.DataFrame(columns=columns)

for feature in data:
    properties = feature.get("properties", {})
    geometry = feature.get("geometry", {})

    date = datetime.utcfromtimestamp(properties.get("time") / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
    magnitude = properties.get("mag")
    latitude = geometry.get("coordinates")[1]
    longitude = geometry.get("coordinates")[0]
    depth = geometry.get("coordinates")[2]
    location = properties.get("place")

    x = x.append({
        "Date": date,
        "Magnitude": magnitude,
        "Latitude": latitude,
        "Longitude": longitude,
        "Depth": depth,
        "Location": location
    }, ignore_index=True)

earthquake_df = pd.concat([df,x],axis=0)

@app.route('/')
def index():
    magnitudes = [5.0 + 0.1 * i for i in range(int((9.5 - 5.0) / 0.1) + 1)]
    time_options = [
        '1 day', '2 days', '1 week', '1 month', '2 months',
        '3 months', '4 months', '5 months', '6 months',
        '7 months', '8 months', '9 months', '10 months', '11 months', '1 year'
    ]

    return render_template('index.html', magnitudes=magnitudes, time_options=time_options)

@app.route('/generate_map', methods=['POST'])
def generate_map():
    # Get user inputs from the form
    lower_magnitude = float(request.form['lower_magnitude'])
    upper_magnitude = float(request.form['upper_magnitude'])
    time_range = request.form['time_range']


    # Filter data based on magnitude and time range
    filtered_df = filter_earthquakes(df, lower_magnitude, upper_magnitude, time_range)

    # Generate the map
    earthquake_map = generate_folium_map(filtered_df)
    map_html = earthquake_map.get_root().render()

    return map_html


def filter_earthquakes(df, lower_magnitude, upper_magnitude, time_range):
    # Convert 'Date' column to datetime type if it's not already
    df = earthquake_df
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter based on magnitude range
    filtered_df = df[(df['Magnitude'] >= lower_magnitude) & (df['Magnitude'] <= upper_magnitude)]

    # Filter based on time range
    if time_range == '1 day':
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
    elif time_range == '2 days':
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
    elif time_range == '1 week':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=1)
    elif time_range == '1 month':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=4)
    elif time_range == '2 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=8)
    elif time_range == '3 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=12)
    elif time_range == '4 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=16)
    elif time_range == '5 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=20)
    elif time_range == '6 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=24)
    elif time_range == '7 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=28)
    elif time_range == '8 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=32)
    elif time_range == '9 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=36)
    elif time_range == '10 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=40)
    elif time_range == '11 months':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=44)
    elif time_range == '1 year':
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=52)
    else:
        # Default to 1 day if an invalid option is selected
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
  
    return filtered_df


@app.route('/generate_future_earthquakes_map', methods=['POST'])
def generate_future_earthquakes_map():
    days_ahead = int(request.form['days_ahead'])

    # Load your random forest model
    with open('rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    df = earthquake_df
    df = pd.read_csv("earthquaketill2023.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    magnitude_threshold = 6.5 #considered the threatening magnitude of earthquakes that can cover medium to large areas

    selected_earthquakes = df[df['Magnitude'] > magnitude_threshold]

    selected_earthquakes['DaysSinceStart'] = (selected_earthquakes['Date'] - selected_earthquakes['Date'].min()).dt.days

    df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
    df = df.dropna(subset=['Depth'])

    features = ['DaysSinceStart', 'Latitude', 'Longitude']
    target = 'Magnitude'
   
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(selected_earthquakes[features], selected_earthquakes[target])
    future_dates = pd.date_range(selected_earthquakes['Date'].max() + pd.DateOffset(1), periods=days_ahead, freq='D')
    future_data = pd.DataFrame({
        'DaysSinceStart': (future_dates - df['Date'].min()).days,
        'Latitude': np.random.uniform(low=-90, high=90, size=days_ahead),  # Example random latitude
        'Longitude': np.random.uniform(low=-180, high=180, size=days_ahead)
    })


    future_data['Magnitude'] = model.predict(future_data[features])
    # Create a folium map
    future_earthquake_map = generate_folium_map(future_data, is_future=True)
    map_html = future_earthquake_map.get_root().render()

    return render_template('future_earthquakes.html', map=map_html)



def generate_folium_map(df, is_future=False):
    # Create a folium map centered at a location of your choice
    earthquake_map = folium.Map(location=[0, 0], zoom_start=2)

    # Add markers for each earthquake
    for index, row in df.iterrows():
        if is_future:
            popup_text = f"Predicted Magnitude: {row['Magnitude']:.2f}"
        else:
            popup_text = f"Magnitude: {row['Magnitude']:.2f}, Date: {row['Date'].strftime('%Y-%m-%d')}"

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup_text,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(earthquake_map)

    return earthquake_map


@app.route('/future_earthquakes')
def future_earthquakes():
    magnitudes = [5.0 + 0.1 * i for i in range(int((9.5 - 5.0) / 0.1) + 1)]

    return render_template('future_earthquakes.html', magnitudes=magnitudes)


@app.route('/latest_earthquakes')
def latest_earthquakes():
    return render_template('latest_earthquakes.html')

@app.route('/time_series', methods=['GET', 'POST'])
def time_series():
    if request.method == 'POST':
        # Get user-selected time range
        time_range = int(request.form.get('time_range'))

        # Filter the DataFrame based on the time range
        filtered_df = earthquake_df.copy()  # You need to define earthquake_df
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
        filtered_df = filtered_df.set_index('Date').last(f'{time_range}D').reset_index()

        # Generate time series plot
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_df['Date'], filtered_df['Magnitude'], marker='o')
        plt.title('Earthquake Time Series')
        plt.xlabel('Date')
        plt.ylabel('Magnitude')
        plt.grid(True)

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        # Encode the image to base64 for embedding in HTML
        encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')

        # Close the plot to release resources
        plt.close()

        # Pass the encoded image to the template
        return render_template('time_series.html', encoded_image=encoded_image)

    return render_template('time_series.html', encoded_image=None)

if __name__ == '__main__':
    app.run(debug=True)
