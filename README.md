---

**Start copying from here ↓**

```
# 🧵 Fabric Property Predictor

A machine learning-based web application that predicts **Stitch Length (SL)** and **Yarn Count (YC)** for textile fabrics based on various fabric properties.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange.svg)

---

## 📋 Overview

This application helps textile industry professionals predict important fabric properties:

| Prediction | Description |
|------------|-------------|
| **SL_ground** | Stitch Length for ground yarn |
| **SL_elastane** | Stitch Length for elastane yarn |
| **YC_ground_Ne** | Yarn Count (Ne) for ground yarn |
| **YC_ground_Denier** | Yarn Count (Denier) for ground yarn |
| **YC_ground_Filament** | Yarn Count (Filament) for ground yarn |
| **YC_elastane_Denier** | Yarn Count (Denier) for elastane yarn |

---

## 🎯 Features

- ✅ Predicts multiple fabric properties simultaneously
- ✅ User-friendly web interface built with Streamlit
- ✅ Machine Learning models trained on real textile data
- ✅ Supports various fabric compositions and types

---

## 🛠️ Input Parameters

| Feature | Description | Example |
|---------|-------------|---------|
| **Fabrication** | Type of fabric construction | Single Jersey, Rib, Interlock |
| **Composition** | Fiber content percentage | 95% Cotton, 5% Elastane |
| **Machine Diameter** | Knitting machine diameter (inches) | 30, 32, 34 |
| **Gauge** | Machine gauge (needles per inch) | 24, 28, 32 |
| **Color** | Fabric color type | White, Melange, Dyed |
| **GSM** | Grams per square meter | 140, 160, 180 |

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Fabric_Property_predictor.git
cd Fabric_Property_predictor
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Fabric_Property_predictor/
│
├── trained_models/
│   ├── random_forest_model_SL_elastane.pkl
│   ├── random_forest_model_SL_ground.pkl
│   ├── random_forest_model_YC_elastane_Denier.pkl
│   ├── random_forest_model_YC_ground_Denier.pkl
│   ├── random_forest_model_YC_ground_Filament.pkl
│   └── random_forest_model_YC_ground_Ne.pkl
│
├── trained_encoders/
│   └── fabrication_label_encoder.pkl
│
├── main.py
├── pipeline.py
├── requirements.txt
├── X_columns.pkl
├── valid_cotton_ne.pkl
├── valid_denier.pkl
└── README.md
```

---

## 🤖 Model Information

- **Algorithm:** Random Forest Regressor
- **Training Data:** ~8,000 samples

---

## 🔧 Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Joblib** - Model serialization

---

## 👤 Author

**Your Name**

- GitHub: [@your_username](https://github.com/your_username)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
```

**Stop copying here ↑**

---

## 📝 How to Create README.md

### In GitHub:
1. Click **"Add file"** → **"Create new file"**
2. Name it: `README.md`
3. Paste the content above
4. Replace `YOUR_USERNAME` and `Your Name` with your details
5. Click **"Commit new file"**

### In PyCharm:
1. Right-click project → **New** → **File**
2. Name: `README.md`
3. Paste content
4. Save

---

**That's it! Just copy the content between "Start copying" and "Stop copying"** 👆
