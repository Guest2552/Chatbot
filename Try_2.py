import mysql.connector
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import openai

# Set OpenAI API key
openai.api_key = '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'  # Replace with your actual key

# 1. Connect to MySQL and Fetch Data
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Gokul&raj2552",
    database="chatbot"
)

cursor = conn.cursor()

cursor.execute("SELECT * FROM users_dataset")
users_data = cursor.fetchall()

cursor.execute("SELECT * FROM transactions_dataset")
transactions_data = cursor.fetchall()

cursor.execute("SELECT * FROM goals_dataset")
goals_data = cursor.fetchall()

# Convert to Pandas DataFrames
users_df = pd.DataFrame(users_data, columns=['user_id', 'name', 'age', 'gender', 'city', 'occupation', 'income_monthly', 'household_size'])
transactions_df = pd.DataFrame(transactions_data, columns=['txn_id', 'user_id', 'date', 'description', 'category', 'amount', 'txn_type'])
goals_df = pd.DataFrame(goals_data, columns=['goal_id', 'user_id', 'goal_description', 'goal_amount', 'amount_saved', 'deadline'])

cursor.close()
conn.close()

# 2. Preprocessing
users_df['gender_encoded'] = users_df['gender'].map({'F': 0, 'M': 1})
scaler = MinMaxScaler()
users_df['scaled_income'] = scaler.fit_transform(users_df[['income_monthly']])
users_df['name'] = users_df['name'].str.title().str.strip()
users_df['city'] = users_df['city'].str.title().str.strip()
users_df['occupation'] = users_df['occupation'].str.title().str.strip()
users_df['income_per_person'] = users_df['income_monthly'] / users_df['household_size']
users_df['age_group'] = pd.cut(users_df['age'], bins=[0, 25, 35, 45, 100], labels=['18-25', '26-35', '36-45', '45+'])
users_df['is_high_income'] = (users_df['income_monthly'] > 60000).astype(int)

goals_df['deadline'] = pd.to_datetime(goals_df['deadline'])
goals_df['goal_progress_pct'] = (goals_df['amount_saved'] / goals_df['goal_amount']) * 100
goals_df['days_left'] = (goals_df['deadline'] - pd.Timestamp.today()).dt.days
goals_df['is_urgent_goal'] = (goals_df['days_left'] < 30).astype(int)
goals_df['is_halfway_there'] = (goals_df['goal_progress_pct'] >= 50).astype(int)

transactions_df['date'] = pd.to_datetime(transactions_df['date'])
transactions_df['category'] = transactions_df['category'].str.title().str.strip()
transactions_df['is_large_txn'] = (transactions_df['amount'] > 10000).astype(int)
transactions_df['day_of_week'] = transactions_df['date'].dt.day_name()
transactions_df['is_weekend_txn'] = transactions_df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

# 3. Monthly Totals
debits = transactions_df[transactions_df['txn_type'] == 'Debit']
credits = transactions_df[transactions_df['txn_type'] == 'Credit']

user_spending = debits.groupby('user_id')['amount'].sum().reset_index(name='total_debit')
user_income = credits.groupby('user_id')['amount'].sum().reset_index(name='total_credit')
user_summary = pd.merge(user_spending, user_income, on='user_id', how='outer').fillna(0)

# 4. Merge All
merged_df = users_df.merge(goals_df, on='user_id').merge(user_summary, on='user_id')
merged_df['net_balance'] = merged_df['total_credit'] - merged_df['total_debit']

final_features_df = merged_df[['user_id', 'goal_progress_pct', 'days_left', 'net_balance', 'income_per_person',
                               'is_high_income', 'is_urgent_goal', 'is_halfway_there']]

# 5. Train Model
X = merged_df[['income_per_person', 'is_high_income', 'is_urgent_goal', 'is_halfway_there']]
y = merged_df['goal_progress_pct']
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# 6. Generate Strategy
def generate_dynamic_strategy(user_id, goal_description, goal_amount, goal_progress_pct, days_left):
    user_data = users_df[users_df['user_id'] == user_id].iloc[0]
    goal_data = goals_df[goals_df['user_id'] == user_id].iloc[0]
    name = user_data['name']

    income = user_data['income_monthly']
    net_balance = merged_df[merged_df['user_id'] == user_id]['net_balance'].values[0]
    expenses = income - net_balance
    saved_amount = (goal_progress_pct / 100) * goal_amount
    remaining_amount = goal_amount - saved_amount

    monthly_savings_required = round((remaining_amount / days_left) * 30, 2) if days_left != 0 else remaining_amount
    potential_savings = income - expenses
    savings_rate = round((potential_savings / income) * 100) if income != 0 else 0

    # ✅ RAG-style context block
    retrieved_context = f"""
User Name: {name}
Income: ₹{income:,.0f}/month
Net Balance: ₹{net_balance:,.0f}
Estimated Expenses: ₹{expenses:,.0f}/month
Goal: {goal_description} (Target ₹{goal_amount:,.0f})
Saved: ₹{saved_amount:,.0f} ({goal_progress_pct}%)
Days Left: {days_left}
Remaining to Save: ₹{remaining_amount:,.0f}
Estimated Monthly Saving Required: ₹{monthly_savings_required:,.0f}
"""

    # 🧠 GPT Prompt with RAG-style structure
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You're a smart financial advisor. Use the provided financial profile to offer a strategic, motivating plan. Be concise and use emojis."
            },
            {
                "role": "user",
                "content": f"""
Here is the user's financial profile:

{retrieved_context}

Now generate a strategy to help them reach the goal of '{goal_description}' within {days_left} days. Include savings advice, income improvement ideas, and any trade-offs they can consider.
"""
            }
        ],
        max_tokens=4096,
        temperature=0.7
    )

    return response['choices'][0]['message']['content'].strip()

# 7. CLI Removed – only used as module in Flask
if __name__ == "__main__":
    print("This script is meant to be imported into a Flask app.")
