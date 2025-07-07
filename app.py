from flask import Flask, render_template, request
from flask_mail import Mail, Message
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from keras.models import load_model
from src.get_data import GetData
from src.utils import create_figure, prediction_from_model
import flask_monitoringdashboard as dashboard
import threading
import time
import os
import logging
from datetime import datetime

app = Flask(__name__)

# Configuration pour Flask-Mail
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.example.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)

# Configuration de la journalisation
logging.basicConfig(filename='app_metrics.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Variable pour compter les appels
call_count = 0
lock = threading.Lock()
THRESHOLD = 100

def reset_call_count():
    global call_count
    while True:
        time.sleep(3600)  # Attendre une heure
        with lock:
            call_count = 0
            logging.info(f"Call count reset at {datetime.now()}")

# Démarrer un thread pour réinitialiser le compteur toutes les heures
threading.Thread(target=reset_call_count, daemon=True).start()

@app.before_request
def count_requests():
    global call_count
    request.start_time = time.time()  # Enregistrer le temps de début de la requête

    with lock:
        call_count += 1
        logging.info(f"Request received at {datetime.now()}. Total calls this hour: {call_count}")
        if call_count > THRESHOLD:
            send_alert_email()

@app.after_request
def log_response_time(response):
    # Calculer le temps de réponse
    response_time = time.time() - request.start_time
    logging.info(f"Request completed in {response_time:.2f} seconds at {datetime.now()}")
    return response

def send_alert_email():
    msg = Message("Alerte: Nombre d'appels dépassé",
                  recipients=['admin@example.com'])
    msg.body = f"L'application a reçu plus de {THRESHOLD} appels en une heure."
    mail.send(msg)

data_retriever = GetData(url="https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-du-trafic-en-temps-reel/exports/json?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B")
data = data_retriever()
model = load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    start_time = time.time()
    logging.info(f"Index function called at {datetime.now()}")

    if request.method == 'POST':
        fig_map = create_figure(data)
        graph_json = fig_map.to_json()
        selected_hour = request.form['hour']
        cat_predict = prediction_from_model(model, selected_hour)
        color_pred_map = {0:["Prédiction : Libre", "green"], 1:["Prédiction : Dense", "orange"], 2:["Prédiction : Bloqué", "red"]}
        response = render_template('index.html', graph_json=graph_json, text_pred=color_pred_map[cat_predict][0], color_pred=color_pred_map[cat_predict][1])
    else:
        fig_map = create_figure(data)
        graph_json = fig_map.to_json()
        response = render_template('index.html', graph_json=graph_json)

    response_time = time.time() - start_time
    logging.info(f"Index function completed in {response_time:.2f} seconds at {datetime.now()}")
    return response
dashboard.bind(app)

if __name__ == '__main__':
    app.run(debug=True)
