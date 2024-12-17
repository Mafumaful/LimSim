import sqlite3
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the database path
PATH = "/Users/miakho/Code/LimSim/database/detector.db"

# Attack type are "None", "ATK_BRK", "ATK_FLT"
def plot_attack_type_data():
    conn = sqlite3.connect(PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM attack_stats")
    data = cur.fetchall()
    conn.close()
    
    time_step = [d[0] for d in data]
    # Extract data
    attack_type = [d[1] for d in data]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["Attack Type"]
    )
    
    # Add the attack type subplot
    fig.add_trace(
        go.Scatter(x=time_step, y=attack_type, mode='lines', name="Attack Type"),
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Attack Type Data",
        height=600,
        showlegend=True
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Frame", row=1, col=1)
    
    # Show plot
    fig.show()
    
plot_attack_type_data()