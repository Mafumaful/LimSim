import sqlite3
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the database path
PATH = "/home/miakho/python_code/LimSim/detector.db"

# Plot the cost data
def plot_cost_data():
    conn = sqlite3.connect(PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM cost_data")
    data = cur.fetchall()
    conn.close()
    
    time_step = [d[0] for d in data]
    # Extract data
    path_cost = [d[1] for d in data]
    traffic_rule_cost = [d[2] for d in data]
    collision_possibility_cost = [d[3] for d in data]
    total_cost = [d[4] for d in data]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Combined Costs (Path, Traffic Rule, Collision)", "Total Cost"]
    )
    
    # Add the combined cost components subplot
    fig.add_trace(
        go.Scatter(x=time_step, y=path_cost, mode='lines', name="Path Cost"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_step, y=traffic_rule_cost, mode='lines', name="Traffic Rule Cost"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_step, y=collision_possibility_cost, mode='lines', name="Collision Possibility Cost"),
        row=1, col=1
    )
    
    # Add the total cost subplot
    fig.add_trace(
        go.Scatter(x=time_step, y=total_cost, mode='lines', name="Total Cost"),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Cost Data Subplots",
        height=600,
        showlegend=True
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Cost Value", row=1, col=1)
    fig.update_yaxes(title_text="Cost Value", row=2, col=1)
    
    fig.show()

plot_cost_data()
