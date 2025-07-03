from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import openai
from datetime import timedelta

from Try_2 import users_df, goals_df, merged_df, generate_dynamic_strategy  # use logic from your script

app = Flask(__name__)
app.secret_key = "admin"
app.permanent_session_lifetime = timedelta(minutes=30)

@app.route('/', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form["name"].strip().lower()
        user_match = users_df[users_df['name'].str.lower().str.strip() == name]
        if not user_match.empty:
            session.permanent = True
            session["user_name"] = name
            session["user_id"] = int(user_match.iloc[0]['user_id'])
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid username")
    return render_template("login.html")

@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    if "user_name" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        goal_description = request.form["goal_description"]
        goal_amount = float(request.form["goal_amount"])
        goal_progress_pct = float(request.form["goal_progress_pct"])
        days_left = int(request.form["days_left"])
        user_id = session["user_id"]

        strategy = generate_dynamic_strategy(user_id, goal_description, goal_amount, goal_progress_pct, days_left)
        return render_template("strategy.html", strategy=strategy)

    return render_template("dashboard.html", user=session["user_name"].title())

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
