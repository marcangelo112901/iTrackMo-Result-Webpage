import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import r2_score

keypoints_2d = pd.read_csv("2d_keypoints_test.csv", index_col=False).to_numpy()
keypoints_3d_predictions = pd.read_csv("3d_keypoints_predictions.csv", index_col=False).to_numpy()
keypoints_3d_ground_truth = pd.read_csv("3d_keypoints_ground_truth.csv", index_col=False).to_numpy()
epoch_data = pd.read_csv("TrainTestLossOverEpoch.csv", index_col=False).to_numpy()
k2d_mean = np.genfromtxt("k2D_Mean.csv", delimiter=",")
k2d_std = np.genfromtxt("k2D_STD.csv", delimiter=",")
k3d_mean = np.genfromtxt("k3D_Mean.csv", delimiter=",").reshape(3)
k3d_std = np.genfromtxt("k3D_STD.csv", delimiter=",").reshape(3)
epoch_data[:, 0] = epoch_data[:, 0] + 1

    
def plot_3d_keypoints(keypoints1, keypoints2):

    skeleton = [
        (0, 1), (0, 2),       # Nose to Eyes
        (1, 3), (2, 4),       # Eyes to Ears
        (5, 6), (5, 7), (7, 9),   # Left Arm
        (6, 8), (8,10),           # Right Arm
        (5,11), (6,12),           # Shoulders to Hips
        (11,13), (13,15),         # Left Leg
        (12,14), (14,16),         # Right Leg
        (11,12)                   # Hip connection
    ]

    # Skeleton 1 (red)
    joints1 = go.Scatter3d(
        x=keypoints1[:,2], y=keypoints1[:,0], z=-keypoints1[:,1],
        mode='markers', marker=dict(size=5, color='red'),
        name='Prediction'
    )
    lines1 = []
    for (i,j) in skeleton:
        lines1.append(go.Scatter3d(
            x=[keypoints1[i,2], keypoints1[j,2]],
            y=[keypoints1[i,0], keypoints1[j,0]],
            z=[-keypoints1[i,1], -keypoints1[j,1]],
            mode='lines', line=dict(color='red', width=3), showlegend=False
        ))

    # Skeleton 2 (blue)
    joints2 = go.Scatter3d(
        x=keypoints2[:,2], y=keypoints2[:,0], z=-keypoints2[:,1],
        mode='markers', marker=dict(size=5, color='blue'),
        name='Ground Truth'
    )
    lines2 = []
    for (i,j) in skeleton:
        lines2.append(go.Scatter3d(
            x=[keypoints2[i,2], keypoints2[j,2]],
            y=[keypoints2[i,0], keypoints2[j,0]],
            z=[-keypoints2[i,1], -keypoints2[j,1]],
            mode='lines', line=dict(color='blue', width=3), showlegend=False
        ))

    # Combine all traces
    fig = go.Figure(data=[joints1] + lines1 + [joints2] + lines2)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')
        ),
        width=700, height=700
    )
    return fig
    

def plot_2d_keypoints(keypoints):

    skeleton = [
        (0, 1), (0, 2),       # Nose to Eyes
        (1, 3), (2, 4),       # Eyes to Ears
        (5, 6), (5, 7), (7, 9),   # Left Arm
        (6, 8), (8,10),           # Right Arm
        (5,11), (6,12),           # Shoulders to Hips
        (11,13), (13,15),         # Left Leg
        (12,14), (14,16),         # Right Leg
        (11,12)                   # Hip connection
    ]

    # Joints
    joints = go.Scatter(
        x=keypoints[:,0], y=keypoints[:,1],
        mode='markers', marker=dict(size=8, color='red'),
        name='Keypoints'
    )

    # Skeleton lines
    lines = []
    for (i,j) in skeleton:
        lines.append(go.Scatter(
            x=[keypoints[i,0], keypoints[j,0]],
            y=[keypoints[i,1], keypoints[j,1]],
            mode='lines', line=dict(color='red', width=2), showlegend=False
        ))

    fig = go.Figure(data=[joints] + lines)
    fig.update_layout(
        xaxis=dict(scaleanchor="y", title="X"),
        yaxis=dict(title="Y", autorange='reversed'),  # reverse Y for image coords
        width=600, height=600
    )
    return fig

st.title("iTrackMo's Custom MLP Model")
tab1, tab2, tab3 = st.tabs(["Model Overview", "Training/Test", "Results"])

with tab1:
    st.markdown("### Model Design Overview")
    st.image("custom_model_overview.png")
    
with tab2:
    st.markdown("### Model Train and Test Result")
    st.markdown(f"Final Train Loss: {epoch_data[-1, 1]:.4f}")
    st.markdown("Final Train Loss without gaussian noise: 0.0251")
    st.markdown(f"Final Test Loss: {epoch_data[-1, 2]:.4f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epoch_data[:, 0],
        y=epoch_data[:, 1],
        mode='lines+markers',
        name='Train Loss',
        hovertemplate='Train Loss: %{y:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=epoch_data[:, 0],
        y=epoch_data[:, 2],
        mode='lines+markers',
        name='Test Loss',
        hovertemplate='Test Loss: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title='Training vs Test Loss over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
with tab3:
    st.markdown("### Model Result")
    st.markdown("Coefficient of Determination (R<sup>2</sup>) Score: 0.8937<br>Mean Per Joint Position Error: 0.229", unsafe_allow_html=True)
    sample_index = st.slider("Select sample index", 0, keypoints_2d.shape[0]-1, 0)
    
    sampleR2 = r2_score(keypoints_3d_predictions[sample_index], keypoints_3d_ground_truth[sample_index])
    mpjpe = np.mean(np.linalg.norm(keypoints_3d_ground_truth[sample_index] - keypoints_3d_predictions[sample_index]))
    st.markdown(f"Sample R<sup>2</sup> Score: {sampleR2:.4f}", unsafe_allow_html=True)
    st.markdown(f"Sample MPJPE: {mpjpe:.4f}")
    
    st.write("2D Keypoints Graph")
    keypoint2d_unprocess = keypoints_2d[sample_index] * k2d_std + k2d_mean
    fig = plot_2d_keypoints(keypoint2d_unprocess.reshape(17,2))
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("3D Keypoints Graph")
    fig = plot_3d_keypoints(keypoints_3d_predictions[sample_index].reshape(17,3) * k3d_std + k3d_mean, keypoints_3d_ground_truth[sample_index].reshape(17,3) * k3d_std + k3d_mean)
    st.plotly_chart(fig, use_container_width=True)