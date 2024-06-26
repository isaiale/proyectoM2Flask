from flask import Flask, request, render_template, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modeloStartClass.pkl')
app.logger.debug('Modelo cargado correctamente.')

# Diccionario para mapear las clases numéricas a etiquetas de texto
class_mapping = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        data = [
            float(request.form['alpha']),
            float(request.form['delta']),
            float(request.form['u']),
            float(request.form['g']),
            float(request.form['r']),
            float(request.form['i'])
        ]
        
        # Crear un DataFrame con los datos
        columns = ['alpha', 'delta', 'u', 'g', 'r', 'i']
        data_df = pd.DataFrame([data], columns=columns)
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        prediction_class = prediction.argmax(axis=-1)[0]
        app.logger.debug(f'Predicción: {prediction_class}')
        
        # Obtener la etiqueta de la clase predicha
        predicted_label = class_mapping[prediction_class]
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': int(prediction_class), 'label': predicted_label})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
