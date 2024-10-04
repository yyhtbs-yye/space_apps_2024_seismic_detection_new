import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_sta_lta_ratio(t, x, sta, lta, sta_lta_ratios, detections=None, draw_detection=False):
    """
    Function to plot the original signal, STA, LTA, and STA/LTA ratios with detections marked.

    Parameters:
    t (numpy array): Time array corresponding to the signal.
    x (numpy array): Original signal.
    sta (numpy array): Short-term average (STA) values.
    lta (numpy array): Long-term average (LTA) values.
    sta_lta_ratios (numpy array): STA/LTA ratio values.
    detections (list): Times where detections are made (based on STA/LTA ratio exceeding the threshold).
    """

    # Create subplots with shared x-axis across 3 rows
    fig_sta_lta = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                vertical_spacing=0.05,  # Adjust spacing between subplots
                                subplot_titles=("Original Signal", "STA and LTA", "STA/LTA Ratio"))

    # Plot the original signal in the first subplot (row=1)
    fig_sta_lta.add_trace(go.Scatter(x=t, 
                                    y=x, 
                                    mode='lines', 
                                    line=dict(color='black', width=1),
                                    name='Original Signal'),
                        row=1, col=1)

    # Plot STA in the second subplot (row=2)
    fig_sta_lta.add_trace(go.Scatter(x=t, 
                                    y=sta, 
                                    mode='lines', 
                                    line=dict(color='blue', width=0.5),
                                    name='STA of the Original Signal'),
                        row=2, col=1)

    # Plot LTA in the second subplot (row=2)
    fig_sta_lta.add_trace(go.Scatter(x=t, 
                                    y=lta, 
                                    mode='lines', 
                                    line=dict(color='red', width=0.5),
                                    name='LTA of the Original Signal'),
                        row=2, col=1)

    # Plot the STA/LTA ratio in the third subplot (row=3)
    fig_sta_lta.add_trace(go.Scatter(x=t, 
                                    y=sta_lta_ratios, 
                                    mode='lines', 
                                    line=dict(color='green', width=0.5),
                                    name='STA/LTA Ratio'),
                        row=3, col=1)

    # Plot detection times as vertical lines in all subplots
    if draw_detection:
        for detection_time in detections:
            fig_sta_lta.add_vline(x=detection_time, line=dict(color='purple', dash='dash'), row='all')

    # Update layout to align x-axes and adjust plot size
    fig_sta_lta.update_layout(
        xaxis_title='time_rel(sec)',
        height=800,  # Adjust the total height of the figure
        width=1000,  # Adjust the width of the figure
        showlegend=True
    )
    fig_sta_lta.update_xaxes(exponentformat = 'power')

    # Update y-axis labels for individual subplots
    fig_sta_lta['layout']['yaxis']['title'] = 'Velocity (m/s)'  # Y-axis for the first subplot
    fig_sta_lta['layout']['yaxis2']['title'] = 'STA/LTA Amplitude'  # Y-axis for the second subplot
    fig_sta_lta['layout']['yaxis3']['title'] = 'STA/LTA Ratio'  # Y-axis for the third subplot

    # Show the figure
    fig_sta_lta.show()
