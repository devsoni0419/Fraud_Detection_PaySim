# 💳 Fraud Detection Using PaySim Dataset

An end-to-end **machine learning project for detecting fraudulent mobile money transactions** using the **PaySim dataset**.  
The project covers data preprocessing, class imbalance handling, model training & evaluation, and an interactive **Streamlit web application** for real-time predictions.

---

## 📌 Project Overview

Fraud detection is a classic **highly imbalanced classification problem**, where fraudulent transactions form a very small fraction of the data.  
This project focuses on building a robust ML pipeline that prioritizes **recall and F1-score**, which are critical in real-world fraud detection systems.

---

## 🗂️ Project Structure

Fraud_Detection_PaySim/
│── app.py # Streamlit web app
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│
├── data/
│ └── paysim.csv # Dataset (not included in repo)
│
└── src/
├── preprocessing.py # Data cleaning & feature engineering
├── train_model.py # Model training
├── evaluate.py # Model evaluation
├── plot_curves.py # ROC / Precision-Recall curves
└── predict.py # Prediction utilities


---

## 📊 Dataset

- **Name:** PaySim – A Financial Mobile Money Simulator  
- **Transactions:** ~6 million  
- **Class Distribution:** Highly imbalanced (fraud ≪ non-fraud)

### 🔗 Dataset Sources
- Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1  
- Research Paper: https://www.sciencedirect.com/science/article/pii/S0377221716308358  

> ⚠️ The dataset is **not included** in this repository due to GitHub’s file size limits.  
> After downloading, place the file at:
data/paysim.csv


---

## ⚙️ Tech Stack

- **Python**
- **pandas, numpy** – data processing
- **scikit-learn** – model training & evaluation
- **imbalanced-learn** – handling class imbalance
- **matplotlib** – visualization
- **joblib** – model saving/loading
- **Streamlit** – interactive web application

---

## 🧠 Machine Learning Workflow

1. **Data Preprocessing**
   - Dropped non-informative identifiers
   - Encoded transaction types
   - Feature scaling where required

2. **Handling Class Imbalance**
   - Used resampling techniques from `imbalanced-learn`

3. **Model Training**
   - Trained classification models suitable for imbalanced data

4. **Evaluation Metrics**
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
git clone https://github.com/devsoni0419/Fraud_Detection_PaySim.git
cd Fraud_Detection_PaySim


### 2️⃣ Create and Activate Virtual Environment
python -m venv myenv
myenv\Scripts\activate


### 3️⃣ Install Dependencies
pip install -r requirements.txt


### 4️⃣ Download Dataset
- Download `paysim.csv` from Kaggle
- Place it inside the `data/` folder

### 5️⃣ Train the Model
python src/train_model.py


### 6️⃣ Run the Streamlit App
streamlit run app.py


---

## 📈 Results

- Successfully detects fraudulent transactions despite extreme class imbalance
- Improved fraud recall using imbalance-handling techniques
- Interactive UI for real-time fraud prediction

---

## 🔮 Future Improvements

- Add SHAP / feature-importance explanations
- Hyperparameter tuning
- Cost-sensitive learning
- Deployment on Streamlit Cloud

---

## 👤 Author

**Dev Soni**  
B.Tech CSE (AI & ML)  
DAV Institute of Engineering & Technology, Jalandhar  

GitHub: https://github.com/devsoni0419  

---

## 📄 License

This project is intended for **educational and research purposes**.
