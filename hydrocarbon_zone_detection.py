import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import seaborn as sns
import streamlit as st
import joblib
import lasio

st.title("Hydrocarbon Zone Detection App")
st.write("Upload well log data and detect potential hydrocarbon-bearing zones using deep learning and Random Forest models.")

uploaded_file1 = st.file_uploader("Upload LAS1 CSV File", type="csv")
uploaded_file2 = st.file_uploader("Upload SLAM CSV File", type="csv")

las_file = st.file_uploader("(Optional) Upload LAS File", type=["las"])

if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    df1.replace(-999.25, np.nan, inplace=True)
    df2.replace(-999.25, np.nan, inplace=True)
    df2.rename(columns={'DEPT': 'DEPTH'}, inplace=True)

    df = pd.merge(df1, df2, on='DEPTH', suffixes=('_slam1', '_slam2'))
    df.dropna(axis=1, thresh=0.5 * len(df), inplace=True)
    df.dropna(inplace=True)

    st.success("Data successfully loaded and merged.")

    if 'HC_ZONE' not in df.columns:
        st.info("No HC_ZONE label found. Generating based on heuristic rule.")
        df['HC_ZONE'] = ((df['GR_slam1'] < 75) & (df['RD_slam1'] > 10) & (df['PE_slam1'] > 2.5)).astype(int)

    st.write("Label Distribution:")
    st.bar_chart(df['HC_ZONE'].value_counts())

    # Select features
    X = df.drop(columns=['DEPTH', 'ZDEN_slam1', 'ZDEN_slam2', 'HC_ZONE'])
    y = df['HC_ZONE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Neural Network
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    y_pred_nn = (model.predict(X_test_scaled) > 0.5).astype(int)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    # Classification Reports
    st.subheader("Classification Reports")
    st.text("Neural Network")
    st.text(classification_report(y_test, y_pred_nn))
    st.text("Random Forest")
    st.text(classification_report(y_test, y_pred_rf))

    # Confusion Matrices
    st.subheader("Confusion Matrices")
    fig_cm, ax_cm = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt='d', ax=ax_cm[0], cmap='Blues')
    ax_cm[0].set_title("NN Confusion Matrix")
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=ax_cm[1], cmap='Greens')
    ax_cm[1].set_title("RF Confusion Matrix")
    st.pyplot(fig_cm)

    # Plot Detected Zones
    st.subheader("Hydrocarbon Zones Over Depth")
    depth_test = df.loc[y_test.index, 'DEPTH']
    fig_depth, ax_depth = plt.subplots(figsize=(6, 10))
    ax_depth.plot(y_test.values, depth_test, label='Actual', linewidth=2)
    ax_depth.plot(y_pred_nn, depth_test, label='Predicted NN', linestyle='--')
    ax_depth.plot(y_pred_rf, depth_test, label='Predicted RF', linestyle=':')
    ax_depth.invert_yaxis()
    ax_depth.set_xlabel("Hydrocarbon Zone")
    ax_depth.set_ylabel("Depth (m)")
    ax_depth.set_title("Zone Detection")
    ax_depth.legend()
    st.pyplot(fig_depth)

    # Export Detected Zones
    st.subheader("Export Detected Zones")
    export_df = df[['DEPTH']].copy()
    export_df['HC_ZONE'] = df['HC_ZONE']
    export_df['Pred_NN'] = 0
    export_df['Pred_RF'] = 0
    export_df.loc[y_test.index, 'Pred_NN'] = y_pred_nn.flatten()
    export_df.loc[y_test.index, 'Pred_RF'] = y_pred_rf

    csv_export = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Zone Predictions CSV", csv_export, file_name="zone_predictions.csv")

    # Real-time Prediction from User Input
    st.subheader("Try Real-Time Prediction")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    nn_result = model.predict(user_scaled)[0][0]
    rf_result = rf.predict(user_scaled)[0]

    st.write(f"**Neural Network Prediction (HC Zone):** {'Yes' if nn_result > 0.5 else 'No'} ({nn_result:.2f})")
    st.write(f"**Random Forest Prediction (HC Zone):** {'Yes' if rf_result == 1 else 'No'}")

    # LAS file preview
    if las_file is not None:
        st.subheader("LAS File Preview")
        las = lasio.read(las_file)
        las_df = las.df().reset_index()
        st.write(las_df.head())
else:
    st.warning("Please upload both LAS1 and SLAM CSV files to proceed.")
