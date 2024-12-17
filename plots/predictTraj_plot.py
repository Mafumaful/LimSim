import sqlite3
import json
import plotly.graph_objects as go

# Define the database path
PATH = "/Users/miakho/Code/LimSim/database/detector.db"

def plot_predict_traj():
    # Connect to the SQLite database
    conn = sqlite3.connect(PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM predict_traj")
    data = cur.fetchall()
    conn.close()

    # Extract unique time steps
    time_steps = [d[0] for d in data]

    # Create a plotly figure
    fig = go.Figure()

    # Initialize frames for animation
    frames = []

    for t in time_steps:
        frame_data = []
        for row in data:
            if row[0] != t:
                continue

            # Extract vehicle data
            vehicle_id = row[1]
            x_pos = row[2]
            y_pos = row[3]
            p_traj = json.loads(row[4])  # Predicted trajectory
            x_traj = [p[0] for p in p_traj]
            y_traj = [p[1] for p in p_traj]

            # Add trajectory line and current position marker for this vehicle
            frame_data.append(
                go.Scatter(
                    x=x_traj, y=y_traj, mode='markers',
                    line=dict(width=2),
                    name=f"Vehicle {vehicle_id} Trajectory",
                    showlegend=(t == time_steps[0])  # Show legend only in the first frame
                )
            )
            frame_data.append(
                go.Scatter(
                    x=[x_pos], y=[y_pos], mode='markers',
                    marker=dict(size=8),
                    name=f"Vehicle {vehicle_id} Position",
                    showlegend=False
                )
            )

        # Add frame for the current time step
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Initialize with the first frame
    if frames:
        for trace in frames[0].data:
            fig.add_trace(trace)

    # Add frames to the figure
    fig.frames = frames

    # Configure layout and controls
    fig.update_layout(
        updatemenus=[
            {
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 50, 'redraw': True},
                                        'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                          'mode': 'immediate',
                                          'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }
        ],
        title="Multi-Vehicle Predicted Trajectories",
        xaxis=dict(
            title="X Position",
            range=[80, 500]  # Fixed x-axis range
        ),
        yaxis=dict(
            title="Y Position",
            range=[-20, 420]  # Fixed y-axis range
        ),
        legend=dict(title="Legend"),
        # Ensure equal scaling for x and y axes
        xaxis_scaleanchor="y",
        xaxis_scaleratio=1
    )

    # Show the animation
    fig.show()

# Call the function
plot_predict_traj()