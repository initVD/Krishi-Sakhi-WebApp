# 🌿 Krishi Sakshi (कृषि साक्षी) - SIH 2025
An AI-powered web application designed to be a digital companion for farmers, providing instant, context-aware agricultural advice.

Smart India Hackathon 2025 - Problem Statement
This project is a solution for the challenge of providing scalable, expert-level farming advice to smallholder farmers in Kerala. The core problem is that generic advisories are ineffective, and farmers often lack access to timely, personalized guidance based on their specific location, crop, weather, and soil conditions.

# ✨ Our Solution: Krishi Sakshi
Krishi Sakhi (meaning "Farming's Friend") is a multi-featured web application that acts as a digital companion for every farmer. It bridges the knowledge gap by combining a fine-tuned computer vision model for disease diagnosis with a powerful Large Language Model (LLM) for answering complex questions, all while being aware of the farmer's unique context.

Our platform empowers farmers with on-demand support, enhances productivity through timely actions, and supports government agricultural departments by automating first-level support.

# ✅ Features Implemented
We have successfully implemented the following core features from the problem statement:

Farmer & Farm Profiling: ✅ Users can register with key details including location, land size, crop, soil type, and irrigation method.

Conversational Interface: ✅ Farmers can interact via text and voice (in Malayalam), making the platform highly accessible.

Activity Tracking: ✅ All diagnoses and questions are automatically saved to a personal Activity Log for future reference.

Personalized Advisory: ✅ The application provides proactive, contextual guidance by leveraging the farmer's real-time location to give weather-based advice and AI-generated pest warnings.

Reminders & Alerts: ✅ A crop calendar system allows farmers to log sowing dates and receive a personalized schedule of timely nudges for crop operations.

Knowledge Engine: ✅ A dual-AI system pulls from a fine-tuned vision model for disease diagnosis and a Large Language Model (Gemini) for broad, expert-level best practices.

# 🛠️ Technology Stack
This project is built with a modern, scalable web stack.

Backend
Python 3

Flask (Web Framework)

TensorFlow Lite (For AI model inference)

Google Gemini (Large Language Model API)

SQLite (Local Database)

APScheduler (For background reminder tasks)

Frontend
HTML5

CSS3

JavaScript (For interactivity, voice recognition, and 3D visuals)

Three.js (For 3D graphics)

# 🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8+

A virtual environment tool (venv)

Git

Installation & Setup
Clone the repository:

 ## Bash

git clone https://github.com/your-username/Krishi-Sakhi-WebApp.git
cd Krishi-Sakhi-WebApp
Create and activate a virtual environment:

## Bash

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install the required libraries:

## Bash

pip install -r requirements.txt
Add your API Keys:

Open the app.py file.

Replace 'YOUR_GEMINI_API_KEY' with your actual Google Gemini API key.

Replace 'YOUR_OPENWEATHERMAP_API_KEY' with your actual OpenWeatherMap API key.

Run the application:

Bash

python app.py
Open your web browser and navigate to http://127.0.0.1:5000.

#


