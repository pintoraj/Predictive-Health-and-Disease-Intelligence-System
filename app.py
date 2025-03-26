import google.generativeai as genai
from flask import Flask, render_template, request, make_response, redirect, url_for, flash
from dotenv import load_dotenv
import os
import datetime
from PIL import Image
import io
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import pickle
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf


load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
# Load the Random Forest CLassifier model
filename = 'Models/diabetes-model.pkl'
filename1 = 'Models/cancer-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
rf = pickle.load(open(filename1, 'rb'))


# --- Gemini API Setup ---
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-flash')

app = Flask(__name__)

# --- Create directories for saving interactions ---
interactions_dir_diagnosis = "diagnosis_database"
if not os.path.exists(interactions_dir_diagnosis):
    os.makedirs(interactions_dir_diagnosis)

interactions_dir_medicine = "medicine_database"
if not os.path.exists(interactions_dir_medicine):
    os.makedirs(interactions_dir_medicine)

interactions_dir_image_analysis = "image_analysis"
if not os.path.exists(interactions_dir_image_analysis):
    os.makedirs(interactions_dir_image_analysis)

interactions_dir_chat_history = "chat_history"
if not os.path.exists(interactions_dir_chat_history):
    os.makedirs(interactions_dir_chat_history)

# --- User Database (SQLite) ---
# Initialize the database connection
db = sqlite3.connect('users.db')
cursor = db.cursor()

# Create the users table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')

# Commit the changes to the database
db.commit()

# Close the database connection
db.close()

# --- Utility Functions ---
def get_gemini_response(prompt):
    """Gets a response from the Gemini API."""
    response = model.generate_content(prompt)
    return response.text.strip()

def save_interaction(user_input, insights, directory):
    """Saves user input and Gemini's response to a text file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"interaction_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("User Input:\n")
        f.write(user_input + "\n\n")
        f.write("Response:\n")
        f.write(insights)

# --- Function to read interactions from files ---
def read_interactions_from_file(directory):
    """Reads interaction files from a directory."""
    interactions = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                interactions.append({"filename": filename, "content": content})
    return interactions

# --- Diagnosis App Functions ---
def get_user_input_from_form():
    gender = request.form.get('gender')
    age = int(request.form.get('age'))
    diabetes = request.form.get('diabetes')
    previous_diseases = request.form.get('previous_diseases')
    choice = request.form.get('choice')
    return gender, age, diabetes, previous_diseases, choice

def get_medical_insights(symptoms=None, disease=None, gender=None, age=None,
                         diabetes=None, previous_diseases=None):
    """Provides medical insights from Gemini."""
    if symptoms:
        prompt = (
            f"A {gender} patient aged {age} presents with these symptoms: {symptoms}. "
            f"They have {'diabetes' if diabetes == 'yes' else 'no diabetes'}"
            f"{' and a history of ' + previous_diseases if previous_diseases != 'none' else ''}. "
            "Provide the following information:\n"
            "1. **Symptom Descriptions:** Describe the provided symptoms in detail.\n"
            "2. **Possible Diseases:** List potential diseases or conditions.\n"
            "3. **Disease Descriptions:** Provide detailed descriptions of the possible diseases.\n"
            "4. **Precautions:** Suggest precautions to take for each possible disease.\n"
            "5. **Medications:** List common medications used to treat each possible disease.\n"
            "6. **Workout:** Recommend suitable workout regimens for each possible disease (if any).\n"
            "7. **Diet:** Provide dietary advice for each possible disease.\n"
            "8. **Suggested Medical Diagnosis:** Recommend relevant diagnostic tests (e.g., blood tests, X-rays)."
        )
    elif disease:
        prompt = (
            f"Provide comprehensive information about {disease}, specifically for a "
            f"{gender} patient aged {age} with "
            f"{'diabetes' if diabetes == 'yes' else 'no diabetes'}"
            f"{' and a history of ' + previous_diseases if previous_diseases != 'none' else ''}."
            "\nInclude the following:\n"
            "1. **Disease Description:** \n"
            "2. **Precautions:** \n"
            "3. **Medications:** \n"
            "4. **Workout:** \n"
            "5. **Diet:** \n"
            "6. **Suggested Medical Diagnosis:** "
        )
    else:
        return "Please provide either symptoms or a disease."

    response = get_gemini_response(prompt)
    return response



# --- Chatbot Function ---
def generate_response(prompt):
    """Generates a text response from the chatbot model."""
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Authentication Decorator (No Session, Basic Redirect) ---
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'username' in request.cookies:  # Check for username cookie
            return func(*args, **kwargs)
        else:
            return redirect(url_for('login'))
    return wrapper

# --- Routes ---

@app.route('/', methods=['GET'])
@login_required
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Hash the password for security
        hashed_password = generate_password_hash(password)

        # Connect to the database
        db = sqlite3.connect('users.db')
        cursor = db.cursor()

        try:
            # Insert the new user into the database
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                           (username, hashed_password))
            db.commit()
            return "<script>alert('User created successfully! You can now log in.'); window.location.href='/login';</script>"  # Redirect using JavaScript
        except sqlite3.IntegrityError:
            # Handle username already exists error
            return "<script>alert('Username already exists'); window.location.href='/signup';</script>"
        finally:
            db.close()

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Connect to the database
        db = sqlite3.connect('users.db')
        cursor = db.cursor()

        try:
            cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            if result:
                hashed_password = result[0]
                if check_password_hash(hashed_password, password):
                    response = redirect(url_for('index'))
                    response.set_cookie('username', username)  # Set username cookie
                    return response
                else:
                    return "<script>alert('Invalid password'); window.location.href='/login';</script>"
            else:
                return "<script>alert('User not found'); window.location.href='/login';</script>"
        finally:
            db.close()

    return render_template('login.html')

@app.route('/logout')
def logout():
    response = redirect(url_for('login'))
    response.set_cookie('username', '', expires=0)  # Clear cookie
    return response
@app.route('/disease')
def home():
    return render_template('index1.html')
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)

            return render_template('d_result.html', prediction=my_prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('diabetes'))


@app.route('/cancer')
def cancer():
    return render_template('cancer.html')


@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    if request.method == 'POST':
        rad = float(request.form['Radius_mean'])
        tex = float(request.form['Texture_mean'])
        par = float(request.form['Perimeter_mean'])
        area = float(request.form['Area_mean'])
        smooth = float(request.form['Smoothness_mean'])
        compact = float(request.form['Compactness_mean'])
        con = float(request.form['Concavity_mean'])
        concave = float(request.form['concave points_mean'])
        sym = float(request.form['symmetry_mean'])
        frac = float(request.form['fractal_dimension_mean'])
        rad_se = float(request.form['radius_se'])
        tex_se = float(request.form['texture_se'])
        par_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smooth_se = float(request.form['smoothness_se'])
        compact_se = float(request.form['compactness_se'])
        con_se = float(request.form['concavity_se'])
        concave_se = float(request.form['concave points_se'])
        sym_se = float(request.form['symmetry_se'])
        frac_se = float(request.form['fractal_dimension_se'])
        rad_worst = float(request.form['radius_worst'])
        tex_worst = float(request.form['texture_worst'])
        par_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smooth_worst = float(request.form['smoothness_worst'])
        compact_worst = float(request.form['compactness_worst'])
        con_worst = float(request.form['concavity_worst'])
        concave_worst = float(request.form['concave points_worst'])
        sym_worst = float(request.form['symmetry_worst'])
        frac_worst = float(request.form['fractal_dimension_worst'])

        data = np.array([[rad, tex, par, area, smooth, compact, con, concave, sym, frac, rad_se, tex_se, par_se, area_se, smooth_se, compact_se, con_se, concave_se,
                          sym_se, frac_se, rad_worst, tex_worst, par_worst, area_worst, smooth_worst, compact_worst, con_worst, concave_worst, sym_worst, frac_worst]])
        my_prediction = rf.predict(data)

        return render_template('c_result.html', prediction=my_prediction)


def ValuePredictor(to_predict_list, size):
    loaded_model = joblib.load('models/heart_model')
    to_predict = np.array(to_predict_list).reshape(1, size)
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/predict_heart', methods=['POST'])
def predict_heart():

    if request.method == 'POST':
        try:
            to_predict_list = request.form.to_dict()
            to_predict_list = list(to_predict_list.values())
            to_predict_list = list(map(float, to_predict_list))
            result = ValuePredictor(to_predict_list, 11)

            if(int(result) == 1):
                prediction = 1
            else:
                prediction = 0

            return render_template('h_result.html', prediction=prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('heart'))


# this function use to predict the output for Fetal Health from given data
def fetal_health_value_predictor(data):
    try:
        # after get the data from html form then we collect the values and
        # converts into 2D numpy array for prediction
        data = list(data.values())
        data = list(map(float, data))
        data = np.array(data).reshape(1, -1)
        # load the saved pre-trained model for new prediction
        model_path = 'Models/fetal-health-model.pkl'
        model = pickle.load(open(model_path, 'rb'))
        result = model.predict(data)
        result = int(result[0])
        status = True
        # returns the predicted output value
        return (result, status)
    except Exception as e:
        result = str(e)
        status = False
        return (result, status)


# this route for prediction of Fetal Health
@app.route('/fetal_health', methods=['GET', 'POST'])
def fetal_health_prediction():
    if request.method == 'POST':
        # geting the form data by POST method
        data = request.form.to_dict()
        # passing form data to castome predict method to get the result
        result, status = fetal_health_value_predictor(data)
        if status:
            # if prediction happens successfully status=True and then pass uotput to html page
            return render_template('fetal_health.html', result=result)
        else:
            # if any error occured during prediction then the error msg will be displayed
            return f'<h2>Error : {result}</h2>'

    # if the user send a GET request to '/fetal_health' route then we just render the html page
    # which contains a form for prediction
    return render_template('fetal_health.html', result=None)


def strokeValuePredictor(s_predict_list):
    '''function to predict the output by data we get
    from the route'''

    model = joblib.load('Models/stroke_model.pkl')
    data = np.array(s_predict_list).reshape(1, -1)
    result = model.predict(data)
    return result[0]


@app.route('/stroke')
def stroke():
    return render_template('stroke.html')


@app.route('/predict_stroke', methods=['POST'])
# this route for predicting chances of stroke
def predict_stroke():

    if request.method == 'POST':
        s_predict_list = request.form.to_dict()
        s_predict_list = list(s_predict_list.values())
        # list to keep the values of the dictionary items of request.form field
        s_predict_list = list(map(float, s_predict_list))
        result = strokeValuePredictor(s_predict_list)

        if(int(result) == 1):
            prediction = 1
        else:
            prediction = 0
        return render_template('st_result.html', prediction=prediction)


def liverprediction(final_features):
    # Loading the pickle file
    model_path = 'Models/liver-disease_model.pkl'
    model = pickle.load(open(model_path, 'rb'))
    result = model.predict(final_features)
    return result[0]


@app.route('/liver')
def liver():
    return render_template('liver.html')


@app.route('/predict_liver', methods=['POST'])
# predicting
def predict_liver_disease():

    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        output = liverprediction(final_features)
        pred = int(output)

        return render_template('liver_result.html', prediction=pred)


@app.route("/malaria", methods=['GET', 'POST'])
def malaria():
    return render_template('malaria.html')


@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredict():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((50, 50))
                img = np.asarray(img)
                img = img.reshape((1, 50, 50, 3))
                img = img.astype(np.float64)

                model_path = "Models/malaria-model.h5"
                model = tf.keras.models.load_model(model_path)
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message=message)
    return render_template('malaria_predict.html', pred=pred)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
@login_required
def diagnosis_index():
    if request.method == 'POST':
        gender, age, diabetes, previous_diseases, choice = get_user_input_from_form()

        if choice == '1':
            symptoms = request.form.get('symptoms')
            insights = get_medical_insights(symptoms=symptoms, gender=gender,
                                             age=age, diabetes=diabetes,
                                             previous_diseases=previous_diseases)
            save_interaction(f"Symptoms: {symptoms}, Gender: {gender}, Age: {age}, "
                             f"Diabetes: {diabetes}, Previous Diseases: {previous_diseases}",
                             insights, interactions_dir_diagnosis)
            return render_template('diagnosis_results.html', insights=insights, symptoms=symptoms,
                                    gender=gender, age=age, diabetes=diabetes,
                                    previous_diseases=previous_diseases) 

        elif choice == '2':
            disease = request.form.get('disease')
            gender = request.form.get('gender')
            age = int(request.form.get('age') or 0)  
            diabetes = request.form.get('diabetes')
            previous_diseases = request.form.get('previous_diseases')

            insights = get_medical_insights(disease=disease, gender=gender,
                                             age=age, diabetes=diabetes,
                                             previous_diseases=previous_diseases)
            save_interaction(f"Disease: {disease}, Gender: {gender}, Age: {age}, "
                             f"Diabetes: {diabetes}, Previous Diseases: {previous_diseases}",
                             insights, interactions_dir_diagnosis)
            return render_template('diagnosis_results.html', insights=insights,
                                   gender=gender, age=age, diabetes=diabetes,
                                   previous_diseases=previous_diseases) 

    return render_template('diagnosis_index.html')



@app.route('/history')
@login_required
def history():
    """Renders the history page."""
    diagnosis_history = read_interactions_from_file(interactions_dir_diagnosis)
    medicine_history = read_interactions_from_file(interactions_dir_medicine)
    image_history = read_interactions_from_file(interactions_dir_image_analysis)
    chatbot_history = read_interactions_from_file(interactions_dir_chat_history)
    return render_template('history.html', 
                           diagnosis_history=diagnosis_history,
                           medicine_history=medicine_history, 
                           image_history=image_history, 
                           chatbot_history=chatbot_history)

@app.route('/history/<interaction_type>/<filename>')
@login_required
def download_interaction(interaction_type, filename):
    """Downloads a specific interaction as a text file."""
    if interaction_type == 'diagnosis':
        directory = interactions_dir_diagnosis
    elif interaction_type == 'medicine':
        directory = interactions_dir_medicine
    elif interaction_type == 'image':
        directory = interactions_dir_image_analysis
    elif interaction_type == 'chat':
        directory = interactions_dir_chat_history
    else:
        return "Invalid interaction type", 404  

    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return "File not found", 404 

    with open(filepath, 'r', encoding="utf-8") as f:
        interaction_content = f.read()

    response = make_response(interaction_content)
    response.headers['Content-Type'] = 'text/plain'  
    response.headers['Content-Disposition'] = f'attachment; filename={filename}' 
    return response

if __name__ == '__main__':
    app.run(debug=True)
