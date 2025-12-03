from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import sqlite3
import uuid
import google.generativeai as genai
import datetime
import requests
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.secret_key = 'a-very-long-and-random-secret-key-that-you-should-change'

# --- API KEY CONFIGURATION ---
# IMPORTANT: PASTE YOUR API KEYS HERE
WEATHER_API_KEY = 'your api key'
GEMINI_API_KEY = 'your api key'

if GEMINI_API_KEY and 'YOUR_GEMINI_API_KEY' not in GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use the correct, latest model name
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    llm_model = None
    print("WARNING: Gemini API Key not configured. AI advisory features will be disabled.")

# --- Folder Setup & Database ---
DB_NAME = 'farmers.db'
def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    # Farmers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS farmers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, phone TEXT UNIQUE NOT NULL,
    location TEXT NOT NULL, crop TEXT NOT NULL, land_size REAL, soil_type TEXT, irrigation TEXT)
    ''')
    # Activities table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS activities (id INTEGER PRIMARY KEY, farmer_phone TEXT NOT NULL, activity_type TEXT NOT NULL,
    content TEXT NOT NULL, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')
    # Crop events table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS crop_events (id INTEGER PRIMARY KEY, farmer_phone TEXT NOT NULL,
    crop TEXT NOT NULL, sowing_date DATE NOT NULL, UNIQUE(farmer_phone, crop))
    ''')
    # Crop schedules table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS crop_schedules (id INTEGER PRIMARY KEY, crop_name TEXT NOT NULL,
    activity TEXT NOT NULL, days_after_sowing INTEGER NOT NULL)
    ''')
    conn.commit()
    conn.close()

def populate_schedules():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    if cursor.execute("SELECT COUNT(*) FROM crop_schedules").fetchone()[0] == 0:
        schedules = [
            ('Rice', 'First Weeding', 20), ('Rice', 'Fertilizer Application', 35), ('Rice', 'Harvesting', 120),
            ('Tomato', 'Staking/Support', 25), ('Tomato', 'First Fertilizer', 30), ('Tomato', 'Harvesting Begins', 70),
            ('Banana', 'Fertilizer (Month 2)', 60), ('Banana', 'De-suckering', 150), ('Banana', 'Harvesting Begins', 300),
            ('Potato', 'First Earthing Up', 25), ('Potato', 'Fertilizer Application', 30), ('Potato', 'Harvesting', 90)
        ]
        cursor.executemany("INSERT INTO crop_schedules (crop_name, activity, days_after_sowing) VALUES (?, ?, ?)", schedules)
        conn.commit()
    conn.close()

init_db()
populate_schedules()

# --- Model Loading ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'finetuned_model.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')
interpreter, labels, input_details, output_details = (None, [], None, None)
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open(LABELS_PATH, 'r') as f:
        labels = f.read().splitlines()
except Exception as e:
    print(f"CRITICAL: Error loading TFLite model or labels: {e}")

def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# --- Background Task for Reminders ---
def check_for_reminders():
    print(f"\n--- Running Daily Reminder Check: {datetime.date.today()} ---")
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    
    cursor.execute("SELECT farmer_phone, crop, sowing_date FROM crop_events")
    events = cursor.fetchall()
    
    for event in events:
        farmer_phone, crop, sowing_date_str = event
        sowing_date = datetime.datetime.strptime(sowing_date_str, '%Y-%m-%d').date()
        
        cursor.execute("SELECT activity, days_after_sowing FROM crop_schedules WHERE crop_name = ?", (crop,))
        schedule_rules = cursor.fetchall()
        
        for rule in schedule_rules:
            activity, days_after = rule
            activity_date = sowing_date + datetime.timedelta(days=days_after)
            
            if activity_date == tomorrow:
                reminder_message = f"REMINDER for farmer {farmer_phone}: Tomorrow is the day for '{activity}' on your {crop} crop."
                print(reminder_message)
    
    conn.close()
    print("--- Reminder Check Finished ---\n")

# --- Web Page Routes ---
@app.route('/')
def home():
    if 'phone' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        phone = request.form['phone']
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farmers WHERE phone = ?", (phone,))
        farmer = cursor.fetchone()
        conn.close()
        if farmer:
            session['phone'] = farmer[2]
            session['name'] = farmer[1]
            session['location'] = farmer[3]
            session['crop'] = farmer[4]
            session['land_size'] = farmer[5]
            session['soil_type'] = farmer[6]
            session['irrigation'] = farmer[7]
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Phone number not found.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        location = request.form['location']
        crop = request.form['crop']
        land_size = request.form.get('land_size')
        soil_type = request.form.get('soil_type')
        irrigation = request.form.get('irrigation')
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO farmers (name, phone, location, crop, land_size, soil_type, irrigation) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, phone, location, crop, land_size, soil_type, irrigation))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="This phone number is already registered.")
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if 'phone' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('diagnose.html', prediction_text='No image selected.')
        
        img_bytes = file.read()
        processed_image = process_image(img_bytes)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = output_data[0]
        max_index = np.argmax(probabilities)
        max_prob = probabilities[max_index]
        
        CONFIDENCE_THRESHOLD = 0.5
        if max_prob > CONFIDENCE_THRESHOLD:
            prediction = labels[max_index].replace("___", " ").replace("_", " ")
            result_text = f"Diagnosis: {prediction} ({max_prob:.2%})"
        else:
            result_text = "Unknown or Not a Plant Leaf"

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO activities (farmer_phone, activity_type, content, response) VALUES (?, ?, ?, ?)",
                       (session['phone'], 'Diagnosis', file.filename, result_text))
        conn.commit()
        conn.close()
        
        return render_template('diagnose.html', prediction_text=result_text)
    return render_template('diagnose.html')

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if 'phone' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        if llm_model is None:
            return render_template('ask.html', llm_answer="LLM is not configured.")
        question = request.form.get('question')
        if not question:
            return render_template('ask.html', llm_answer="Please ask a question.")

        location = session.get('location', 'N/A')
        crop = session.get('crop', 'N/A')
        prompt = f"You are Krishi Sakhi, an expert AI assistant for farmers in Kerala, India. Provide a clear, concise, and helpful answer. Farmer's Context: Location: {location}, Main Crop: {crop}. Farmer's Question: \"{question}\"\nAnswer:"
        
        try:
            response = llm_model.generate_content(prompt)
            answer = response.text
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO activities (farmer_phone, activity_type, content, response) VALUES (?, ?, ?, ?)",
                           (session['phone'], 'Question', question, answer))
            conn.commit()
            conn.close()
        except Exception as e:
            answer = "Sorry, I could not process your request at the moment."
        
        return render_template('ask.html', llm_answer=answer)
    return render_template('ask.html')

@app.route('/my_farm', methods=['GET', 'POST'])
def my_farm():
    if 'phone' not in session: return redirect(url_for('login'))
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if request.method == 'POST':
        crop = request.form['crop']
        sowing_date = request.form['sowing_date']
        cursor.execute("INSERT OR REPLACE INTO crop_events (farmer_phone, crop, sowing_date) VALUES (?, ?, ?)",
                       (session['phone'], crop, sowing_date))
        conn.commit()

    cursor.execute("SELECT crop, sowing_date FROM crop_events WHERE farmer_phone = ?", (session['phone'],))
    events = cursor.fetchall()
    
    schedules = {}
    for crop, sowing_date_str in events:
        sowing_date = datetime.datetime.strptime(sowing_date_str, '%Y-%m-%d').date()
        cursor.execute("SELECT activity, days_after_sowing FROM crop_schedules WHERE crop_name = ?", (crop,))
        schedule_rules = cursor.fetchall()
        
        calculated_schedule = []
        for activity, days in schedule_rules:
            activity_date = sowing_date + datetime.timedelta(days=days)
            calculated_schedule.append({'activity': activity, 'date': activity_date.strftime('%d %B, %Y')})
        schedules[crop] = calculated_schedule

    conn.close()
    return render_template('my_farm.html', schedules=schedules)

@app.route('/activity_log')
def activity_log():
    if 'phone' not in session: return redirect(url_for('login'))
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, activity_type, content, response FROM activities WHERE farmer_phone = ? ORDER BY timestamp DESC",
                   (session['phone'],))
    activities = cursor.fetchall()
    conn.close()
    
    return render_template('activity_log.html', activities=activities)

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if 'phone' not in session: return redirect(url_for('login'))
    
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('diagnose.html', prediction_text='No image selected.')
        
        # --- Vision Model Prediction (No Changes Here) ---
        img_bytes = file.read()
        processed_image = process_image(img_bytes)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = output_data[0]
        max_index = np.argmax(probabilities)
        max_prob = probabilities[max_index]
        
        CONFIDENCE_THRESHOLD = 0.5
        prediction_text = ""
        care_instructions = None
        
        if max_prob > CONFIDENCE_THRESHOLD:
            predicted_label = labels[max_index]
            disease_name = predicted_label.replace("___", " ").replace("_", " ")
            prediction_text = f"Diagnosis: {disease_name} ({max_prob:.2%})"

            # --- NEW: Use AI to get care instructions ---
            if llm_model:
                try:
                    # Create a specific prompt for the Gemini LLM
                    prompt = f"""
                    You are an expert agricultural assistant. A farmer has identified the plant disease '{disease_name}'.
                    Provide a clear, step-by-step list of care instructions to treat this disease.
                    Include both organic and chemical treatment options if available.
                    Format the response for easy reading.
                    """
                    response = llm_model.generate_content(prompt)
                    care_instructions = response.text
                except Exception as e:
                    print(f"Gemini Care Instructions Error: {e}")
                    care_instructions = "Could not fetch care instructions from the AI assistant at this time."
            else:
                care_instructions = "AI assistant is not configured. Cannot fetch care instructions."

            # Log the activity
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO activities (farmer_phone, activity_type, content, response) VALUES (?, ?, ?, ?)",
                           (session['phone'], 'Diagnosis', file.filename, prediction_text))
            conn.commit()
            conn.close()
        else:
            prediction_text = "Unknown or Not a Plant Leaf"
        
        return render_template('diagnose.html', prediction_text=prediction_text, care_instructions=care_instructions)

    return render_template('diagnose.html')

@app.route('/get_advisory', methods=['POST'])
def get_advisory():
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')

    if not lat or not lon:
        return jsonify({'error': 'Location not provided'}), 400

    weather_advisory = "Weather data unavailable."
    description = "clear sky" # Default description
    
    if WEATHER_API_KEY and 'YOUR_OPENWEATHERMAP_API_KEY' not in WEATHER_API_KEY:
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        try:
            weather_res = requests.get(weather_url).json()
            if weather_res.get("cod") == 200:
                main = weather_res['main']
                weather = weather_res['weather'][0]
                temp = main['temp']
                description = weather['description']
                
                weather_advisory = f"🌦️ Weather: Current temperature is {temp}°C with {description}."
                if 'rain' in description or 'storm' in description:
                    weather_advisory += "\nRecommendation: Avoid spraying pesticides or applying fertilizer today."
        except Exception as e:
            print(f"Weather API Error: {e}")

    pest_advisory = ""
    if llm_model:
        location = session.get('location', 'N/A')
        crop = session.get('crop', 'N/A')
        prompt = f"""
        Based on the current weather ({description}) in {location} for a farmer growing {crop},
        what is one proactive pest or disease warning you can give for today?
        Keep the advice short and actionable (1-2 sentences).
        """
        try:
            response = llm_model.generate_content(prompt)
            pest_advisory = f"\n\n🦟 AI Advisory: {response.text}"
        except Exception as e:
            print(f"Gemini Pest Advisory Error: {e}")

    full_advisory = weather_advisory + pest_advisory
    return jsonify({'advisory': full_advisory})

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_for_reminders, 'cron', hour=8)
    scheduler.start()
    print("Background scheduler for reminders has been started.")
    app.run(debug=True, use_reloader=False)
