# Personalized Health System

## Overview
The **Personalized Health System** is a comprehensive healthcare platform designed to predict and provide insights for various medical conditions. The system leverages advanced machine learning models and integrates AI-driven insights for accurate diagnosis, medical recommendations, and patient care.

## Features
✅ **Disease Prediction Models:** Predicts potential medical conditions such as:
- Diabetes
- Cancer
- Heart Disease
- Fetal Health Conditions
- Liver Disease
- Stroke
- Malaria Detection via Image Analysis

✅ **Medical Insights:** Provides detailed information on symptoms, treatments, and preventive measures.  
✅ **User Authentication:** Secure login and signup system with password encryption.  
✅ **Interactive Diagnosis System:** AI-powered medical insights for both symptoms and disease-specific queries.  
✅ **Historical Data Management:** Tracks previous diagnoses, medical interactions, and chat history.  
✅ **Image Analysis Support:** Predicts malaria infection by analyzing uploaded images.  

---

## Tech Stack
- **Python** (Flask Framework)
- **TensorFlow/Keras** (Deep Learning Models)
- **Scikit-learn** (Machine Learning Models)
- **Google Gemini API** (For AI-driven medical insights)
- **SQLite** (Database Management)
- **HTML/CSS/JS** (Front-end for UI)

---

## Project Structure
```
├── Models
│   ├── diabetes-model.pkl
│   ├── cancer-model.pkl
│   ├── heart_model.pkl
│   ├── fetal-health-model.pkl
│   ├── liver-disease_model.pkl
│   ├── stroke_model.pkl
│   └── malaria-model.h5
│
├── templates
│   ├── index.html
│   ├── diabetes.html
│   ├── cancer.html
│   ├── heart.html
│   ├── liver.html
│   ├── stroke.html
│   ├── malaria.html
│   ├── result_pages (prediction result pages)
│   └── ...
│
├── diagnosis_database
├── medicine_database
├── image_analysis
├── chat_history
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Personalized-Health-System.git
   cd Personalized-Health-System
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   .\venv\Scripts\activate   # For Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
Create a `.env` file and add your **GOOGLE_API_KEY** for Gemini API integration:
```
GOOGLE_API_KEY=your_api_key_here
```

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the Application**
Visit `http://localhost:5000` in your browser.

---

## Usage
1. **Sign Up / Log In**
2. Navigate to different prediction modules for conditions like:
   - Diabetes Prediction
   - Cancer Detection
   - Heart Disease Analysis
3. Input your medical details to receive predictions and insights.
4. Access your **history** of previous predictions for reference.

---

## Models Used
- **Random Forest Classifier** (Diabetes & Cancer Predictions)
- **Logistic Regression** (Heart Disease Prediction)
- **Deep Learning (CNN)** (Malaria Image Analysis)
- **Decision Trees, KNN, and Other Algorithms** for condition-specific predictions.

---

## Future Enhancements
🔹 Integration of **IoT Devices** for real-time data collection.  
🔹 Expanding medical coverage with more disease prediction models.  
🔹 Enhanced **data visualization** for medical insights.  

---

## Contributing
Contributions are welcome! Feel free to fork the repository, create a branch, and submit a pull request.  

---

## License
This project is licensed under the **MIT License**.

---


Feel free to reach out for queries or collaboration!

