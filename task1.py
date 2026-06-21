import pandas as pd
data = pd.read_csv("house_prices.csv")


X = data[['sqft_living', 'bedrooms', 'bathrooms']]
y = data['price']
# Split Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
# Train Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
model = LinearRegression()
model.fit(X_train, y_train)
# Test Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred) * 100
print("r2 Score:", round(accuracy, 2), "%")

# User Input
sqft = float(input("\nEnter House Area (sqft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = float(input("Enter Number of Bathrooms: "))

# Create DataFrame for Prediction
new_house = pd.DataFrame({
    'sqft_living': [sqft],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms]
})
# Predict Price
predicted_price = model.predict(new_house)
print("\n===== HOUSE PRICE PREDICTION =====")
print("Predicted House Price: ", predicted_price)

