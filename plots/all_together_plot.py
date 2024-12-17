import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the database path
PATH = "/Users/miakho/Code/LimSim/database/detector.db"

def combined_attack_cost_plot():
    # Connect to the SQLite database
    conn = sqlite3.connect(PATH)
    cur = conn.cursor()

    # Fetch data from attack_stats table
    cur.execute("SELECT * FROM attack_stats")
    attack_data = cur.fetchall()
    time_step_attack = [row[0] for row in attack_data]
    attack_type = [row[1] for row in attack_data]

    # Fetch data from cost_data table
    cur.execute("SELECT * FROM cost_data")
    cost_data = cur.fetchall()
    time_step_cost = [row[0] for row in cost_data]
    path_cost = [row[1] for row in cost_data]
    traffic_rule_cost = [row[2] for row in cost_data]
    collision_possibility_cost = [row[3] for row in cost_data]
    total_cost = [row[4] for row in cost_data]

    # Close the database connection
    conn.close()

    # Create subplots: 3 rows, 1 column
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            "Attack Type Over Time",
            "Combined Costs (Path, Traffic Rule, Collision)",
            "Total Cost Over Time"
        ]
    )

    # Add Attack Type trace
    fig.add_trace(
        go.Scatter(
            x=time_step_attack,
            y=attack_type,
            mode='lines+markers',
            name="Attack Type",
            line=dict(color='firebrick')
        ),
        row=1, col=1
    )

    # Add Path Cost trace
    fig.add_trace(
        go.Scatter(
            x=time_step_cost,
            y=path_cost,
            mode='lines',
            name="Path Cost",
            line=dict(color='blue')
        ),
        row=2, col=1
    )

    # Add Traffic Rule Cost trace
    fig.add_trace(
        go.Scatter(
            x=time_step_cost,
            y=traffic_rule_cost,
            mode='lines',
            name="Traffic Rule Cost",
            line=dict(color='green')
        ),
        row=2, col=1
    )

    # Add Collision Possibility Cost trace
    fig.add_trace(
        go.Scatter(
            x=time_step_cost,
            y=collision_possibility_cost,
            mode='lines',
            name="Collision Possibility Cost",
            line=dict(color='orange')
        ),
        row=2, col=1
    )

    # Add Total Cost trace
    fig.add_trace(
        go.Scatter(
            x=time_step_cost,
            y=total_cost,
            mode='lines',
            name="Total Cost",
            line=dict(color='purple')
        ),
        row=3, col=1
    )

    # Update layout with titles and axis labels
    fig.update_layout(
        title="Combined Attack and Cost Data Analysis",
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update x-axis for the bottom subplot
    fig.update_xaxes(title_text="Frame", row=3, col=1)

    # Update y-axes for each subplot
    fig.update_yaxes(title_text="Attack Type", row=1, col=1)
    fig.update_yaxes(title_text="Cost Value", row=2, col=1)
    fig.update_yaxes(title_text="Cost Value", row=3, col=1)

    # Show the combined plot
    fig.show()

# Call the function to display the plot
combined_attack_cost_plot()
