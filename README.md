🏡 USA Housing Price Prediction using Multiple Linear Regression
📘 Project Overview

This project demonstrates Multiple Linear Regression on the USA_Housing.csv dataset to predict house prices based on various features such as area income, house age, number of rooms, bedrooms, and population.
The model is implemented in Python using scikit-learn, and the analysis covers:

Model fitting with all features (except Address)

Model fitting with selected 3 features

Performance comparison using error metrics

Finding the best random_state value for maximum accuracy

Feature normalization (scaling)

🧮 Dataset Information

Dataset Name: USA_Housing.csv
Features:

Avg. Area Income

Avg. Area House Age

Avg. Area Number of Rooms

Avg. Area Number of Bedrooms

Area Population

Address (excluded from model)

Target Variable:

Price

⚙️ Technologies Used

Python 3.x

Libraries:

pandas

numpy

scikit-learn

matplotlib (optional for visualization)

🧠 Project Tasks
Q1: Fit Multiple Linear Regression (All features except “Address”)

Fitted model on all features except Address.

Found coefficients (β₀, β₁, β₂, ... βₖ).

Evaluated model using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Q2: Fit Multiple Linear Regression (Any 3 features except “Address”)

Selected 3 features:
Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms.

Computed coefficients and evaluated using the same metrics.

Compared performance with full-feature model.

Q3: Find Best Random State (0–199)

Performed 200 different random train-test splits.

Selected the random_state that produced highest R² Score for:

All features model

3-features model

Q4: Feature Normalization (Standardization)

Applied StandardScaler to normalize features before fitting the model.

Re-evaluated metrics to check improvement in model performance.

📊 Evaluation Metrics Example
Metric	Without Normalization	With Normalization
MAE	81291.74	81283.50
RMSE	101324.45	101320.12
R² Score	0.9179	0.9181

(Values may vary based on dataset split)

🚀 How to Run the Project

Clone the repository
https://github.com/meet-2621/USA-Housing-Price-Prediction-MLR

Navigate to the folder

cd USA-Housing-Regression


Install dependencies

pip install -r requirements.txt

Run Jupyter Notebook or Python script

jupyter notebook :- Multiple_Linear_Regression_Analysis_USA_Housing(http://localhost:8889/notebooks/Multiple_Linear_Regression_Analysis_USA_Housing.ipynb)


or

python usa_housing_regression.py

🏆 Results Summary

The full-feature model achieved the highest accuracy (R² ≈ 0.92).

Normalization improved model stability.

The best random state (based on R²) ensures reproducibility of results.

The model can predict housing prices effectively using multiple regression.

📁 Repository Structure
USA-Housing-Regression/
│
├── USA_Housing.csv
├── USA_Housing_MLR.ipynb
├── usa_housing_regression.py
├── README.md
└── requirements.txt

🧾 Author

👩‍💻 Manmeet Kaur
MCA | Data Science & Machine Learning Enthusiast
📍 TIET , Patiala
🔗 LinkedIn (https://www.linkedin.com/in/manmeet-kaur-245a372ba/)

💻 GitHub (https://github.com/meet-2621)
