import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import os
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

st.set_page_config(page_title="Smart Fraud Engine UI", layout="wide")

@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    model_path = os.path.join(current_dir, "fraud_pipeline.joblib")
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    elif os.path.exists("fraud_pipeline.joblib"):
        return joblib.load("fraud_pipeline.joblib")
    return None

pipeline = load_assets()

if pipeline is None:
    st.error("⚠️ Missing compiled model. Please run `train.py` first to generate `fraud_pipeline.joblib`.")
else:
    st.title("🛡️ Credit Card Fraud Monitoring & Investigation Deck")
    st.caption(f"Real-Time Behavioral Analysis & Compliance Auditing • Powered by: **{pipeline.get('model_name', 'Custom Model')}**")
    
    @st.cache_data
    def get_dashboard_data():
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        data_path = os.path.join(current_dir, "credit_card_fraud_10k.csv")
        
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        elif os.path.exists("credit_card_fraud_10k.csv"):
            return pd.read_csv("credit_card_fraud_10k.csv")
        return None

    raw_data = get_dashboard_data()
    
    if raw_data is None:
        st.warning("⚠️ Please ensure `credit_card_fraud_10k.csv` is in the same folder as this script.")
    else:
        # Pre-calculate scores globally
        drop_cols = [c for c in ['transaction_id', 'is_fraud'] if c in raw_data.columns]
        X_raw = raw_data.drop(columns=drop_cols)
        
        X_trans = pipeline['preprocessor'].transform(X_raw)
        X_trans = pd.DataFrame(X_trans, columns=pipeline['feature_names'])
        
        raw_probs = pipeline['model'].predict_proba(X_trans)[:, 1]
        risk_scores = np.round(raw_probs * 100, 1)
        
        view_df = raw_data.copy()
        view_df['Risk Score'] = risk_scores
        
        # Navigation Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Live Alerts & Deep Dive", 
            "Interactive Fraud Explorer", 
            "📈 Model Performance",
            "📁 Batch Prediction"
        ])
        
        with tab1:
            st.sidebar.header("Alert Settings")
            optimal_thresh_pct = int(pipeline.get('optimal_threshold', 0.15) * 100)
            alert_level = st.sidebar.slider(
                "Minimum Risk Level to Trigger Alert", 
                0, 100, optimal_thresh_pct,
                help="Threshold used to flag active alerts in the queue."
            )
            
            alert_queue = view_df[view_df['Risk Score'] >= (alert_level)].sort_values(by='Risk Score', ascending=False)
            
            st.subheader(f"⚠️ Live Suspected Transaction Queue ({len(alert_queue)} active matches)")
            st.dataframe(alert_queue[['transaction_id', 'amount', 'merchant_category', 'Risk Score']], use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 Deep Dive & Explainability Modeler")
            
            if not alert_queue.empty:
                selected_id = st.selectbox("Select Transaction to Audit:", alert_queue['transaction_id'].unique())
                
                if selected_id:
                    tx_row = raw_data[raw_data['transaction_id'] == selected_id].iloc[0]
                    tx_features = pd.DataFrame([tx_row]).drop(columns=drop_cols)
                    tx_proc = pipeline['preprocessor'].transform(tx_features)
                    tx_proc = pd.DataFrame(tx_proc, columns=pipeline['feature_names'])
                    
                    score_metric = view_df[view_df['transaction_id'] == selected_id]['Risk Score'].values[0]
                    
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        st.metric(
                            "Continuous Risk Score", 
                            f"{score_metric} / 100", 
                            delta="HIGH THREAT" if score_metric >= optimal_thresh_pct else "REVIEW", 
                            delta_color="inverse"
                        )
                        st.write("**Transaction Metadata:**")
                        st.dataframe(tx_features.T.rename(columns={0: 'Value'}), use_container_width=True)
                        
                    with c2:
                        model = pipeline['model']
                        model_name = pipeline.get('model_name', 'Unknown')
                        
                        try:
                            if 'Forest' in model_name or 'Tree' in model_name:
                                explainer = shap.TreeExplainer(model)
                                raw_shap = explainer.shap_values(tx_proc)
                                if isinstance(raw_shap, list):
                                    val_array = raw_shap[1][0]
                                elif len(raw_shap.shape) == 3:
                                    val_array = raw_shap[0, :, 1]
                                else:
                                    val_array = raw_shap[0]
                            else:
                                explainer = shap.LinearExplainer(model, X_trans)
                                raw_shap = explainer.shap_values(tx_proc)
                                val_array = raw_shap[0]
                                
                            shap_summary = pd.DataFrame({
                                'Attribute': pipeline['feature_names'],
                                'Impact Weight': val_array
                            }).sort_values(by='Impact Weight', key=abs, ascending=False).head(5)
                            
                            fig = px.bar(shap_summary, x='Impact Weight', y='Attribute', orientation='h',
                                         title="What drove this score? (SHAP Values)",
                                         color='Impact Weight', color_continuous_scale=px.colors.sequential.Reds)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"SHAP visualization unavailable for this model configuration. Error: {e}")

        with tab2:
            st.subheader("📊 Interactive Fraud Explorer & Dataset Class Distribution")
            
            # Sub-section for Class Distribution as requested
            col_dist1, col_dist2 = st.columns([1, 1])
            with col_dist1:
                st.write("##### Overall Dataset Class Distribution")
                class_counts = raw_data['is_fraud'].value_counts().reset_index()
                class_counts.columns = ['Class', 'Count']
                class_counts['Class'] = class_counts['Class'].map({0: 'Legitimate (0)', 1: 'Fraudulent (1)'})
                fig_pie = px.pie(class_counts, names='Class', values='Count', 
                                 title="Ratio of Legitimate vs Fraudulent Transactions",
                                 color='Class', color_discrete_map={'Legitimate (0)': '#2ecc71', 'Fraudulent (1)': '#e74c3c'})
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_dist2:
                st.write("##### Explorer Filters")
                merch_filter = st.multiselect("Filter by Merchant", view_df['merchant_category'].unique())
                risk_band = st.slider("Risk Band Range", 0, 100, (0, 100))
                
            filtered_df = view_df[view_df['Risk Score'].between(risk_band[0], risk_band[1])]
            if merch_filter:
                filtered_df = filtered_df[filtered_df['merchant_category'].isin(merch_filter)]
                
            fig_scatter = px.scatter(
                filtered_df, x='transaction_hour', y='amount', color='Risk Score', 
                size='velocity_last_24h', hover_data=['transaction_id', 'merchant_category'],
                title="Transaction Amount vs. Hour colored by Risk Score"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab3:
            st.subheader("📈 Model Performance & Evaluation Metrics")
            st.write(f"Currently evaluated using the trained **{pipeline.get('model_name', 'Model')}** architecture.")
            
            # Re-run evaluation on test split to display live metrics & confusion matrix
            df_sorted = raw_data.sort_values(by=['transaction_hour']).reset_index(drop=True)
            X_eval = df_sorted.drop(columns=[c for c in ['transaction_id', 'is_fraud'] if c in df_sorted.columns])
            y_eval = df_sorted['is_fraud']
            
            val_idx = int(len(df_sorted) * 0.85)
            X_test, y_test = X_eval.iloc[val_idx:], y_eval.iloc[val_idx:]
            
            X_test_proc = pipeline['preprocessor'].transform(X_test)
            X_test_proc = pd.DataFrame(X_test_proc, columns=pipeline['feature_names'])
            
            test_probs = pipeline['model'].predict_proba(X_test_proc)[:, 1]
            optimal_thresh = pipeline.get('optimal_threshold', 0.15)
            test_preds = (test_probs >= optimal_thresh).astype(int)
            
            precision_pr_test, recall_pr_test, _ = precision_recall_curve(y_test, test_probs)
            test_pr_auc = auc(recall_pr_test, precision_pr_test)
            
            acc = accuracy_score(y_test, test_preds)
            prec = precision_score(y_test, test_preds, zero_division=0)
            rec = recall_score(y_test, test_preds, zero_division=0)
            f1 = f1_score(y_test, test_preds, zero_division=0)
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Test PR-AUC", f"{test_pr_auc:.4f}")
            m2.metric("Test Accuracy", f"{acc * 100:.2f}%")
            m3.metric("Precision", f"{prec:.4f}")
            m4.metric("Recall", f"{rec:.4f}")
            m5.metric("F1-Score", f"{f1:.4f}")
            
            st.markdown("---")
            st.write("#### Confusion Matrix on Unseen Test Set")
            cm = confusion_matrix(y_test, test_preds)
            
            # Modern Plotly Express Heatmap (fixes the figure_factory AttributeError)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale='Reds',
                labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
                x=['Predicted Legitimate', 'Predicted Fraud'],
                y=['Actual Legitimate', 'Actual Fraud']
            )
            fig_cm.update_layout(title_text="<b>Confusion Matrix</b>", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        with tab4:
            st.subheader("📁 Batch Prediction & Scoring")
            st.markdown("Upload a CSV file containing transactions matching your feature schema to generate batch risk scores.")
            
            uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
            
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:", batch_df.head(3))
                
                if st.button("Run Batch Inference"):
                    try:
                        # Ensure features align
                        batch_X = batch_df.drop(columns=[c for c in ['transaction_id', 'is_fraud'] if c in batch_df.columns])
                        batch_proc = pipeline['preprocessor'].transform(batch_X)
                        batch_proc = pd.DataFrame(batch_proc, columns=pipeline['feature_names'])
                        
                        batch_probs = pipeline['model'].predict_proba(batch_proc)[:, 1]
                        batch_df['Risk Score (%)'] = np.round(batch_probs * 100, 1)
                        batch_df['Flagged Fraud'] = (batch_probs >= optimal_thresh).astype(int)
                        
                        st.success("✅ Batch scoring completed successfully!")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Download button
                        csv_data = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Scored Results as CSV",
                            data=csv_data,
                            file_name="scored_transactions.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error processing batch file: {e}")