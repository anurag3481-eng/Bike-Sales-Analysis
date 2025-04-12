import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
da=pd.read_csv("D:/Python_Project/bike_sales_india.csv")
#print(da.describe()) #stastical Analysis of Dataset
#print(da.info())#Structure of Dataset
#print(da.isnull().sum())#Return Sum of NA in each column
data=da.fillna({'Avg Daily Distance (km)':da['Avg Daily Distance (km)'].mean(),
                   'Brand':da['Brand'].mode()[0],
                   'Model':da['Model'].mode()[0],
                   'Fuel Type':da['Fuel Type'].mode()[0],
                   'Mileage (km/l)':da['Mileage (km/l)'].mean(),
                   'Owner Type':da['Owner Type'].mode()[0],
                   'Insurance Status':da['Insurance Status'].mode()[0],
                   'Seller Type':da['Seller Type'].mode()[0],
                   'Resale Price (INR)':da['Resale Price (INR)'].mean(),
                   'City Tier':da['City Tier'].mode()[0]})

print(data.isnull().sum())
# Feature Engineering using NumPy and Pandas
# 1. Age of the bike
data['Bike Age'] = 2025 - data['Year of Manufacture']
# 2. Is New (binary flag: newly manufactured bikes)
data['Is_New'] = np.where(data['Bike Age'] <= 1, 1, 0)
# 3. Is Expired Insurance (binary flag)
data['Is_Insurance_Expired'] = np.where(data['Insurance Status'] == 'Expired', 1, 0)
# 4. Price per CC (cost per engine capacity unit)
data['Price_per_CC'] = data['Price (INR)'] / data['Engine Capacity (cc)']
# 5. Resale Ratio: How much value the bike retains
data['Resale_Ratio'] = data['Resale Price (INR)'] / data['Price (INR)']
# 6. Avg KM per Year = Avg Daily Distance * 365 / Bike Age
data['Avg_KM_per_Year'] = np.where(data['Bike Age'] > 0, data['Avg Daily Distance (km)'] * 365 / data['Bike Age'], 0)



#EDA



#Graph-1(bar chart for state-wise sales count)
plt.figure(figsize=(12, 6))
data['State'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Bike Sales by State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Number of Bikes Sold', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Graph-2 (Average Daily Distance by Brand)
brand_avg_distance = data.groupby('Brand')['Avg Daily Distance (km)'].mean().sort_values()
plt.figure(figsize=(12, 6))
plt.plot(brand_avg_distance.index, brand_avg_distance.values, marker='o')
plt.xticks(rotation=45)
plt.title('Average Daily Distance by Brand')
plt.xlabel('Brand')
plt.ylabel('Average Daily Distance (km)')
plt.grid(True)
plt.tight_layout()
plt.show()

#Graph-3 (Pie chart of brand market share)
data['Brand'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90,
    cmap='Pastel1',
    wedgeprops={'linewidth': 1, 'edgecolor': 'black'}
)
plt.title('Brand Market Share', fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.show()

#Graph-4 (Distribution of Bike Prices)
plt.figure(figsize=(12, 6))
sns.histplot(data['Price (INR)'], kde=True, bins=40)
plt.title("Distribution of Bike Prices")
plt.xlabel("Price (INR)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#Graph-5(Resale Ratio vs Bike Age)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Bike Age', y='Resale_Ratio', data=data)
plt.title("Resale Ratio vs Bike Age")
plt.xlabel("Bike Age (Years)")
plt.ylabel("Resale Ratio")
plt.tight_layout()
plt.show()

#Graph-6(Average Resale Price by Brand)
plt.figure(figsize=(10, 5))
data.groupby('Brand')['Resale Price (INR)'].mean().sort_values().plot(kind='barh')
plt.title("Average Resale Price by Brand")
plt.xlabel("Average Resale Price (INR)")
plt.ylabel("Brand")
plt.tight_layout()
plt.show()

#Graph-7(Heatmap of Feature Correlations)
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

#Graph-8(Count occurrences of each Insurance Status)
insurance_counts = data['Insurance Status'].value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.plot(insurance_counts.index, insurance_counts.values, marker='o')
plt.title("Count of Bikes by Insurance Status")
plt.xlabel("Insurance Status")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

#Graph-9(Sales Trend by Registration Year)
plt.figure(figsize=(10, 5))
data['Registration Year'].value_counts().sort_index().plot(kind='line', marker='o',color='red')
plt.title('Sales Trend by Year', fontsize=16)
plt.xlabel('Registration Year', fontsize=12)
plt.ylabel('Number of Bikes Registered', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Graph-10(Bike Price Distribution by City Tier)
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='City Tier', y='Price (INR)', palette='Set2')
plt.title('Bike Price Distribution by City Tier', fontsize=16)
plt.xlabel('City Tier', fontsize=12)
plt.ylabel('Price (INR)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#Graph-11(Corelation between Price(INR) and ResalePricc(INR))
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Price (INR)', y='Resale Price (INR)',palette='tab20')
plt.title('Price vs Resale Price')
plt.xlabel('Price (INR)')
plt.ylabel('Resale Price (INR)')
plt.grid(True)
plt.tight_layout()
plt.show()

#Graph-12(grouped bar plot that shows how different fuel types have varied in count over the years)
fuel_counts = data.groupby(['Year of Manufacture', 'Fuel Type']).size().reset_index(name='Count')
plt.figure(figsize=(12, 6))
sns.barplot(data=fuel_counts, x='Year of Manufacture', y='Count', hue='Fuel Type',palette='tab20')
plt.title('Count of Fuel Types Over Years')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Group-13(PairPlot for Analysis of Selected Numeric columns)
numeric_cols = [
    'Price (INR)','Resale Price (INR)', 'Price_per_CC',
    'Avg Daily Distance (km)', 'Avg_KM_per_Year'
]

data_plot = data[numeric_cols + ['Year of Manufacture']]
sns.pairplot(data_plot, hue='Year of Manufacture', palette='tab10', diag_kind='kde')
plt.suptitle("Pairplot of Bike Features Colored by Year of Manufacture", y=1.02)
plt.show()

#Graph-14(Price by Brand)
plt.figure(figsize=(10, 5))
data.groupby('Brand')['Price (INR)'].mean().sort_values().plot(kind='barh')
plt.title("Average Price by Brand")
plt.xlabel("Average Price (INR)")
plt.ylabel("Brand")
plt.tight_layout()
plt.show()

#Outliers in given dataset
numeric_cols = [
    'Price (INR)', 'Engine Capacity (cc)', 'Mileage (km/l)',
    'Resale Price (INR)', 'Bike Age', 'Price_per_CC', 'Resale_Ratio',
    'Avg Daily Distance (km)', 'Avg_KM_per_Year'
]
outliers=[]
for column in numeric_cols:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers.extend(data[(data[column]<lower_bound)|(data[column]>upper_bound)].index)
print(outliers)

# # Select only the relevant columns for our analysis
X = data[['Price (INR)']]
y = data['Resale Price (INR)']

# Normalize the data using MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
# Reshape y for scaling
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Convert back to DataFrame for easier handling
X_scaled_df = pd.DataFrame(X_scaled, columns=['Normalized Price'])
y_scaled_df = pd.DataFrame(y_scaled, columns=['Normalized Resale Price'])

# Print the normalized data
print("\nFirst 5 rows of normalized data:")
normalized_data = pd.concat([X_scaled_df, y_scaled_df], axis=1)
print(normalized_data.head().round(2))

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled, y_scaled, alpha=0.5)
plt.title('Normalized Price vs Normalized Resale Price')
plt.xlabel('Normalized Price')
plt.ylabel('Normalized Resale Price')
plt.grid(True)
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.05, random_state=50)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the model coefficients
print("\nModel Coefficients:")
print(f"Slope (Coefficient): {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.5, label='Testing Data')
plt.plot(X_scaled, model.predict(X_scaled), color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: Normalized Price vs Normalized Resale Price')
plt.xlabel('Normalized Price')
plt.ylabel('Normalized Resale Price')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print("\nModel Evaluation:")
print(f"Mean Squared Error (Training): {mse_train:.2f}")
print(f"Mean Squared Error (Testing): {mse_test:.2f}")

# Define a clean function to predict resale price for a single input
def predict_resale_price(price):
    # Create a DataFrame with the same column name used during fitting
    price_df = pd.DataFrame([[price]], columns=['Price (INR)'])
    
    # Normalize the input
    price_normalized = scaler_X.transform(price_df)
    
    # Make prediction (in normalized scale)
    resale_normalized = model.predict(price_normalized)
    
    # Convert prediction back to original scale
    resale_predicted = scaler_y.inverse_transform(resale_normalized)[0][0]
    
    # Round to 2 decimal places
    return round(resale_predicted, 2)

# Example usage of the function with a single value
print("\nSingle prediction example:")
input_price =252816

predicted_price = predict_resale_price(input_price)
print(f"Original Price: {input_price:,.2f} INR")
print(f"Predicted Resale Price: {predicted_price:,.2f} INR")

# Interactive prediction
try:
    user_price = float(input("\nEnter a price to predict resale value (INR): "))
    user_prediction = predict_resale_price(user_price)
    print(f"For a bike with original price {user_price:,.2f} INR")
    print(f"The predicted resale price is {user_prediction:,.2f} INR")
except ValueError:
    print("Please enter a valid number")
