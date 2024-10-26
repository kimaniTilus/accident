import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Rd_Accident.csv')

data['Weather'] = data['Weather'].astype('category').cat.codes
data['Time'] = data['Time'].astype('category').cat.codes
data['Road_Type'] = data['Road_Type'].astype('category').cat.codes
data['Road_Condition'] = data['Road_Condition'].astype('category').cat.codes

X = data[['Weather', 'Time', 'Road_Type', 'Vehicle_Speed', 'Driver_Age', 'Num_Vehicles', 'Road_Condition']]
y = data['Accident_Severity']

if X.isnull().any().any() or y.isnull().any():
    print("Warning: There are missing values in the dataset. Please handle them before training.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'accident_severity_model.pkl')

hypothetical_data = pd.DataFrame([[1, 2, 0, 50, 30, 2, 1]],
                                   columns=['Weather', 'Time', 'Road_Type', 'Vehicle_Speed', 'Driver_Age', 'Num_Vehicles', 'Road_Condition'])
predicted_severity = model.predict(hypothetical_data)

print(f'Predicted Accident Severity: {predicted_severity[0]}')

# Evaluate the model's performance
y_pred = model.predict(X_test)

if len(y_test) > 1:
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(f'R^2 Score: {r2_score(y_test, y_pred)}')
else:
    print("Not enough samples to calculate R^2 Score.")