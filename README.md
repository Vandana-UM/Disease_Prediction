🫀 Heart Disease Prediction Toolkit
       Predict the risk of heart disease using Machine Learning! This project leverages Random Forest, Decision Tree, and Logistic Regression models to provide quick predictions based on health parameters.

📖 Project Overview
    Heart disease is one of the leading causes of death worldwide. This project helps predict the likelihood of heart disease based on patient data using AI/ML models.

Features:
    User-friendly input interface using Streamlit
    Predicts risk using Random Forest, Decision Tree, and Logistic Regression
    Provides visualizations of correlations and feature importance
    ROC curves for model evaluation

Dataset:
    Source: Heart Disease UCI dataset on Kaggle
    Features: Age, Sex, Chest Pain, Blood Pressure, Cholesterol, Max Heart Rate, etc.
    Target: condition (0 = No heart disease, 1 = Heart disease)

🛠️ Technologies & Libraries Used
    Python 3
    Pandas, NumPy
    Matplotlib, Seaborn
    Scikit-learn (ML models & preprocessing)
    Streamlit (Web app)
    Pyngrok (for hosting in Colab)
    Pickle (model & scaler serialization)

📂 Project Structure
    Heart_Disease_Prediction/
    │
    ├── Disease_Prediction_Project.ipynb  # Jupyter Notebook with EDA & modeling
    ├── app.py                           # Streamlit app for predictions
    ├── rf_model.pkl                      # Trained Random Forest model
    ├── scaler.pkl                        # Scaler for feature normalization
    ├── requirements.txt                  # Python dependencies
    └── README.md                         # Project documentation

🚀 How to Run
    Locally
      Clone the repository:
        git clone https://github.com/Vandana-UM/Disease_Prediction.git
        cd Disease_Prediction
      Install dependencies:
        pip install -r requirements.txt
      Run the Streamlit app:
        streamlit run app.py
        Open the URL shown in the terminal to interact with the app.
        On Google Colab
        Upload your notebook and dataset.
        Run all cells to train models and save the rf_model.pkl & scaler.pkl.
      Run the Streamlit app and use ngrok for public access:
          from pyngrok import ngrok
          public_url = ngrok.connect(port='8501')
          public_url
⚠️ Note: Ngrok now requires a verified account and authtoken.

📊 Visualizations Included
      Heatmap: Feature correlations
      Countplot: Heart disease cases by sex
      Scatterplot: Age vs Max Heart Rate
      Cluster map: Correlation clustering
      ROC Curves: Model performance comparison

💡 Next Steps / Improvements
      Deploy the app on Heroku or Streamlit Cloud for public access.
      Add more ML models like XGBoost or SVM for higher accuracy.
      Include interactive dashboards for advanced analytics.
      Integrate user authentication for personalized health tracking.

📈 Model Performance (Current)
      Model	Accuracy	Precision	Recall	F1 Score
      Logistic Regression	0.73	0.70	0.75	0.72
      Decision Tree	0.77	0.73	0.79	0.76
      Random Forest	0.70	0.67	0.71	0.69
⚡ Contact
Developed by Vandana-UM
GitHub: Vandana-UM
