# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np

# Load 'best_performance.csv' dataset
best_performance = pd.read_csv('recent_performance.csv')

# Handling Missing Values
best_performance.fillna({'WinPercentageLast5': 0, 'ManagerExperienceLast5': 0}, inplace=True)


# Remove '%' signs from 'WinPercentageLast5' and convert to float
best_performance['WinPercentageLast5'] = best_performance['WinPercentageLast5'].str.rstrip('%').astype(float)

# Handle 'Win' and 'Draw' values by setting them to 0
best_performance['WinPercentageLast5'] = best_performance['WinPercentageLast5'].replace(['Win', 'Draw'], '0').astype(float)


# Load 'teams_ranking.csv' dataset
teams_ranking = pd.read_csv('teams_ranking.csv')

# Handling Missing Values (fill with zeros in this example)
teams_ranking.fillna(0, inplace=True)

# Rename columns to remove spaces and use underscores
teams_ranking.columns = teams_ranking.columns.str.replace(' ', '_')

# Merge the two datasets based on a common identifier (e.g., 'HomeTeam' column)
merged_data = pd.merge(best_performance, teams_ranking, left_on=['HomeTeam'], right_on=['Team'], how='inner')
merged_data.drop(columns=['Team'], inplace=True)  # Remove duplicate 'Team' column

# Exclude rows with 'Draw' as the outcome
merged_data = merged_data[merged_data['Outcome'] != 'Draw']

# Encode 'Outcome' column (assuming you want to use it as the target)
# Label encoding for categorical outcomes
label_encoder = LabelEncoder()
merged_data['Outcome'] = label_encoder.fit_transform(merged_data['Outcome'])



# Data Preprocessing
# - Handle missing values
# - Create time-based features from 'MatchDate'

# Convert 'MatchDate' to datetime format
merged_data['MatchDate'] = pd.to_datetime(merged_data['MatchDate'])

# Extract relevant date-based features
merged_data['MatchYear'] = merged_data['MatchDate'].dt.year
merged_data['MatchMonth'] = merged_data['MatchDate'].dt.month
merged_data['MatchDay'] = merged_data['MatchDate'].dt.day

# One-hot encode team names (categorical variables)
team_encoder = LabelEncoder()
team_columns = ['HomeTeam', 'AwayTeam']
merged_data[team_columns] = merged_data[team_columns].apply(team_encoder.fit_transform)

# Define features (X) and target (y)
X = merged_data.drop(['Outcome', 'MatchDate'], axis=1)
y = merged_data['Outcome']

# Split the data into training and testing sets (hold-out validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Hyperparameter Tuning (Random Forest example)
model = RandomForestClassifier(random_state=42)  # You can tune hyperparameters as needed

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Other evaluation metrics (classification report, confusion matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Betting Strategy (customize this part based on your strategy)
def betting_strategy(predictions, actual_results, betting_odds):
    # Implement your betting strategy here
    # Example: Always bet on 'Home' team if predicted probability of 'Home' win > threshold
    threshold = 0.6
    bets = []
    for i in range(len(predictions)):
        if predictions[i] == 'Home' and betting_odds[i]['Home'] > threshold:
            bets.append('Home')
        elif predictions[i] == 'Away' and betting_odds[i]['Away'] > threshold:
            bets.append('Away')
        else:
            bets.append('No Bet')
    return bets

# Simulate betting and calculate ROI
betting_odds = merged_data[['HomeOdds', 'AwayOdds']].values.tolist()
bet_results = betting_strategy(y_pred, y_test, betting_odds)

# Calculate returns based on betting strategy
initial_balance = 1000  # Initial betting balance
balance = initial_balance
bet_amount = 100  # Fixed bet amount for each bet

for i in range(len(bet_results)):
    if bet_results[i] == y_test.iloc[i]:
        balance += bet_amount * (betting_odds[i][y_test.iloc[i]] - 1)  # Assuming decimal odds
    else:
        balance -= bet_amount

# Calculate ROI
roi = (balance - initial_balance) / initial_balance * 100

print(f"Final Balance: ${balance:.2f}")
print(f"ROI: {roi:.2f}%")
