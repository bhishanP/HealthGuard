# HealthGuard

A modern web application for predicting hospital readmission risk in diabetic patients using machine learning.

---

## 🚀 Project Overview

**HealthGuard** leverages a trained ML model to predict patient outcomes based on clinical data. It features a user-friendly web interface and a REST API for seamless predictions.

---

## ✨ Features

- **Web Interface:** Responsive, modern form for patient data entry
- **REST API:** `/predict` endpoint for programmatic access
- **Machine Learning:** Uses a pre-trained model (`model.pkl`)
- **Data Mapping:** Human-readable mappings for categorical features

---

## 🗂️ File Structure

- `app.py` — Flask app, loads the ML model, exposes the `/predict` API
- `templates/index.html` — Beautiful web form for patient data input
- `model.pkl` — Pre-trained machine learning model
- `diabetic_data.csv` — Dataset used for model training (large)
- `IDS_mapping.csv` — Mappings for categorical features
- `readmission.ipynb` — Jupyter notebook for data exploration/modeling (large)

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.7+
- Flask
- pandas
- joblib

### Installation

```bash
git clone https://github.com/bhishanP/HealthGuard.git
cd HealthGuard
pip install flask pandas joblib
```

### Running the App

```bash
python app.py
```

Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## 🛠️ API Usage

- **Endpoint:** `/predict`
- **Method:** POST
- **Content-Type:** `application/json`
- **Request Body:** JSON object with patient features (see form fields in `index.html`)
- **Response:** JSON with prediction result

#### Example Request
```json
{
  "encounter_id": 12345,
  "gender": "Male",
  "age": 55,
  "admission_type_id": 1,
  "discharge_disposition_id": 1,
  "num_procedures": 2,
  "num_medications": 10,
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 0,
  "diag_2": "250.01",
  "nateglinide": "No",
  "pioglitazone": "Yes",
  "rosiglitazone": "No",
  "acarbose": "No",
  "miglitol": "No",
  "troglitazone": "No",
  "tolazamide": "No",
  "examide": "No",
  "citoglipton": "No",
  "glyburide-metformin": "No",
  "metformin-rosiglitazone": "No"
}
```

---

## 📊 Data Dictionary

- **diabetic_data.csv:** Patient records and features for prediction
- **IDS_mapping.csv:** Maps categorical IDs (admission type, discharge disposition, etc.) to descriptions

---

## 📝 Customization

- Retrain or update the model using `readmission.ipynb`
- Update the web form (`index.html`) to match any changes in model input features

---

## 📄 License

Specify your license here (e.g., MIT, Apache 2.0).

---
