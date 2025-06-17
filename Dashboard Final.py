import streamlit as st
import pandas as pd
import joblib
import os
import math
import altair as alt
import webbrowser
import matplotlib.pyplot as plt
from sklearn import tree
from interpret import show
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
import tempfile
import os
import interpret
from interpret import preserve
import subprocess
import threading
import webbrowser
import streamlit.components.v1 as components
import time

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret import show
set_visualize_provider(InlineProvider())

import streamlit.components.v1 as components

from interpret.provider import InlineProvider
from interpret import set_visualize_provider
from interpret import show
import streamlit.components.v1 as components
import streamlit.components.v1 as components

set_visualize_provider(InlineProvider())
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

st.set_page_config(page_title="EBM Dashboard", layout="wide")
st.title("🔍 Explainable Machine Learning Visualization")
st.markdown("""
<style>
/* Remove the default progress bar background */
.stSlider [data-baseweb="slider"] > div > div {
    background: transparent !important;
}

/* Remove the default filled color on the slider (it will not be visible anymore) */
.stSlider [data-baseweb="slider"] > div {
    background: transparent !important;
}

/* Add a custom blue line track */
.stSlider [data-baseweb="slider"] > div::before {
    content: "";
    position: absolute;
    top: 50%;
    left: 0;
    width: 100%;
    height: 2px;
    background-color: #007bff;  /* Blue line */
    transform: translateY(-50%);
    z-index: 1;
}

/* Style the slider handle (dot) */
.stSlider [role="slider"] {
    background-color: orange !important;
    border: 2px solid #333 !important;
    width: 16px !important;
    height: 16px !important;
    border-radius: 50% !important;
    position: relative;
    z-index: 2;  /* Ensure the dot stays on top */
}
</style>
""", unsafe_allow_html=True)

if "show_explanations" not in st.session_state:
    st.session_state["show_explanations"] = False

# --------------------------
# 🔁 Dynamic model loader
# --------------------------
def load_model(model_choice, triage_level, escalation_type, simplified = False):
    model_map = {
        "Logistic Regression": "logreg",
        "Support Vector Machine": "svm",
        "Decision Tree": "DT",
        "Explainable Boosting Machine (EBM)": "ebm"
    }

    model_key = model_map.get(model_choice)
    escalation_formatted = escalation_type.replace("-", "").title()
    if simplified:
        # Construct filename for simplified model
        filename = f"{model_key}_simplified({triage_level}, '{escalation_formatted}').pkl"
    else:
        filename = f"{model_key}({triage_level}, '{escalation_formatted}').pkl"
    st.write(f"📂 Loading model file: `{filename}`")
    model_path = os.path.join("Modellen", filename)

    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model file not found: {filename}")
        st.stop()

# --------------------------
# 📅 Load dataset for local explanation
# --------------------------
def load_dataset(triage_level, escalation_type):
    escalation_formatted = escalation_type.replace("-", "").title()
    filename = f"dataset({triage_level}, '{escalation_formatted}')"
    dataset_path = os.path.join("Datasets", filename)

    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    else:
        st.warning(f"Dataset file not found: {filename}")
        return None
import re
def add_edge_labels(dot_source, model):
    lines = dot_source.splitlines()
    tree_ = model.tree_
    edge_lines = []

    for i in range(tree_.node_count):
        left_child = tree_.children_left[i]
        right_child = tree_.children_right[i]

        if left_child != right_child:  # not a leaf
            edge_lines.append(f'{i} -> {left_child} [label="yes"];')
            edge_lines.append(f'{i} -> {right_child} [label="no"];')

    # Replace existing edge lines with labeled ones
    new_lines = []
    for line in lines:
        if "->" in line:
            continue  # skip original unlabeled edges
        new_lines.append(line)

    return '\n'.join(new_lines + edge_lines)

def simplify_feature_name(fn: str) -> str:
    """
    If fn contains a substring like 'Q_1234', return just 'Q_1234'.
    Otherwise, return fn unchanged.
    """
    m = re.search(r'(Q_\d+)', fn)
    return m.group(1) if m else fn

# … inside your Streamlit block …

def get_combined_auc_scores():
    files = {
        "Logistic Regression": "logreg_resultaten.csv",
        "Support Vector Machine": "SVM_resultaten.csv",
        "Decision Tree": "DT_resultaten.csv",
        "Explainable Boosting Machine (EBM)": "EBM_resultaten.csv",
    }

    all_rows = []
    for model_name, filename in files.items():
        file_path = os.path.join("Modellen", filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                auc = row.get("ROC_AUC_Score", None)
                fit_quality = (
                    "poor" if auc < 0.6 else
                    "mediocre" if auc < 0.7 else
                    "good"
                ) if pd.notnull(auc) else "no_data"

                all_rows.append({
                    "Model": model_name,
                    "Triage Level": int(row["Triage_MINDD"]),
                    "Escalation Type": row["Escalatie"].strip().title(),
                    "AUC ROC Score": f"{auc:.3f}" if pd.notnull(auc) else "N/A",
                    "Fit Quality Tag": fit_quality
                })
        else:
            all_rows.append({
                "Model": model_name,
                "Triage Level": "N/A",
                "Escalation Type": "N/A",
                "AUC ROC Score": "Missing File",
                "Fit Quality Tag": "missing"
            })

    df = pd.DataFrame(all_rows)
    return df.drop(columns=["Fit Quality Tag"]), df["Fit Quality Tag"]


# --------------------------
# 📊 Paginated coefficient/importance plot
# --------------------------
def plot_coefficients_paged(df, title, slider_key):
    total = len(df)
    page_size = 20
    num_pages = math.ceil(total / page_size)

    if total == 0:
        st.info(f"No data to display for: {title}")
        return

    page = st.slider(f"{title} — Page", 1, num_pages, 1, key=slider_key)
    start = (page - 1) * page_size
    end = start + page_size
    slice_df = df.iloc[start:end]

    chart = alt.Chart(slice_df).mark_bar().encode(
        x=alt.X('Value:Q'),
        y=alt.Y('Feature:N', sort='-x'),
        tooltip=['Feature', 'Value']
    ).properties(height=500, title=f"{title} (Page {page} of {num_pages})")

    st.altair_chart(chart, use_container_width=True)

# --------------------------
# 📌 Model Selection View
# --------------------------
set_visualize_provider(InlineProvider())

if not st.session_state["show_explanations"]:
    st.write("### 🔧 Select Parameters")

    triage_level = st.selectbox("Select MINDD Triage Level", options=[1, 2, 3, 4, 5])
    escalation_type = st.radio("Select Escalation Type", options=["Escalation", "De-Escalation"])
    model_choice = st.selectbox("Select a Model Type", options=[
        "Logistic Regression",
        "Support Vector Machine",
        "Decision Tree",
        "Explainable Boosting Machine (EBM)"
    ])

    st.session_state["triage_level"] = triage_level
    st.session_state["escalation_type"] = escalation_type
    st.session_state["model_choice"] = model_choice

    # ⬆️ Generate Explanations button moved here
    if st.button("Generate Explanations"):
        model_map = {
            "Logistic Regression": "logreg",
            "Support Vector Machine": "svm",
            "Decision Tree": "DT",
            "Explainable Boosting Machine (EBM)": "ebm"
        }
        model_key = model_map.get(model_choice)
        escalation_formatted = escalation_type.replace("-", "").title()
        filename = f"{model_key}({triage_level}, '{escalation_formatted}').pkl"
        model_path = os.path.join("Modellen", filename)

        if os.path.exists(model_path):
            st.session_state["show_explanations"] = True
            st.rerun()
        else:
            st.warning(f"Model file not found: {filename}. Please check your selection.")

    # 📈 Now display the model performance overview
    st.write("### 📈 Model Performance Overview (AUC ROC Scores)")

    auc_df, fit_tags = get_combined_auc_scores()

    def color_row_by_fit_quality(tag):
        color_map = {
            "good": "lightgreen",
            "mediocre": "#FFFACD",
            "poor": "#FFCCCB",
            "no_data": "#f0f0f0",
            "missing": "#f0f0f0"
        }
        return [f"background-color: {color_map.get(tag, 'white')}"] * auc_df.shape[1]

    row_styles = pd.DataFrame([color_row_by_fit_quality(tag) for tag in fit_tags], columns=auc_df.columns)
    styled_auc_df = auc_df.style.apply(lambda _: row_styles, axis=None)

    st.dataframe(styled_auc_df, use_container_width=True, height=400)

    st.markdown("#### 🎨 Legend")
    st.markdown("""
    <div style='line-height: 1.6'>
    <span style='background-color: lightgreen; padding: 4px 8px; border-radius: 4px;'>Good fit (AUC ≥ 0.7)</span>  
    <span style='background-color: #FFFACD; padding: 4px 8px; border-radius: 4px;'>Mediocre fit (0.6 ≤ AUC &lt; 0.7)</span>  
    <span style='background-color: #FFCCCB; padding: 4px 8px; border-radius: 4px;'>Poor fit (AUC &lt; 0.6)</span>  
    <span style='background-color: #f0f0f0; padding: 4px 8px; border-radius: 4px;'>No data / missing file</span>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.simplified = False

else:
    if st.button("🔙 Back to Selection"):
        st.session_state["show_explanations"] = False
        st.rerun()

    st.write("### 🔎 Global and Local Explanations")

    model_choice = st.session_state.get("model_choice", "Explainable Boosting Machine (EBM)")
    triage_level = st.session_state.get("triage_level", 1)
    escalation_type = st.session_state.get("escalation_type", "Escalation")

    # Add button to simplify model
    if st.button("Simplify Model"):
        st.session_state.simplified = True
        st.success("Simplified model loaded.")
        st.rerun()
    
    model = load_model(model_choice, triage_level, escalation_type, st.session_state.simplified)


    def display_auc_score(model_name, file_name, triage_level, escalation_type):
        if st.session_state.get("simplified", False):
            if "_resultaten.csv" in file_name:
                file_name = file_name.replace("_resultaten.csv", "_simplified_resultaten.csv")

        aspects_path = os.path.join("Modellen", file_name)
        
        if os.path.exists(aspects_path):
            aspects_df = pd.read_csv(aspects_path)
            aspect_row = aspects_df[
                (aspects_df['Triage_MINDD'].astype(int) == int(triage_level)) &
                (aspects_df['Escalatie'].str.strip().str.lower() == escalation_type.strip().lower())
            ]

            if not aspect_row.empty:
                auc_score = aspect_row.iloc[0]['ROC_AUC_Score']
                if auc_score < 0.6:
                    auc_color = "red"
                    fit_text = "Poor fit"
                elif 0.6 <= auc_score < 0.7:
                    auc_color = "yellow"
                    fit_text = "Mediocre fit"
                else:
                    auc_color = "green"
                    fit_text = "Good fit"

                st.markdown(f"### {model_name} AUC ROC Score: {auc_score:.3f}")
                st.markdown(
                    f'<div style="background-color:{auc_color}; color:white; padding:10px; border-radius:5px; text-align:center;">'
                    f'{fit_text}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"No matching AUC score found in '{file_name}'.")
        else:
            st.error(f"Could not find '{file_name}'.")

    if model_choice == "Logistic Regression":
        display_auc_score("Logistic Regression", "logreg_resultaten.csv", triage_level, escalation_type)
    elif model_choice == "Decision Tree":
        display_auc_score("Decision Tree", "DT_resultaten.csv", triage_level, escalation_type)
    elif model_choice == "Support Vector Machine":
        display_auc_score("SVM", "SVM_resultaten.csv", triage_level, escalation_type)
    elif model_choice == "Explainable Boosting Machine (EBM)":
        display_auc_score("EBM", "EBM_resultaten.csv", triage_level, escalation_type)

    if model_choice == "Explainable Boosting Machine (EBM)":
        if model is not None:
            global_exp = model.explain_global()
            feature_names = global_exp.feature_names
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': global_exp.data()['scores']
            }).sort_values(by='Value', ascending=False).reset_index(drop=True)
            directional_scores = []
            for name, score in zip(model.term_names_, model.term_scores_):
                    directional_scores.append((name, score[1]))

            st.write("### 📊 Global Feature Importance")

            # Calculate relative importance
            top_20_sum = importance_df.head(20)['Value'].sum()
            total_sum = importance_df['Value'].sum()
            relative_importance = (top_20_sum / total_sum) * 100

            st.markdown(f"**Top 20 Features contribute:** `{relative_importance:.2f}%` of total importance")

            plot_coefficients_paged(importance_df, "Global Importance", "ebm_global")
        else:
            st.warning("EBM model not loaded.")
    elif model_choice in ["Logistic Regression", "Support Vector Machine"]:
        if model is not None:
            try:
                if hasattr(model, 'coef_'):
                    coef = model.coef_[0]
                    feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"Feature {i}" for i in range(len(coef))]
                    coef_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': coef
                    })

                    coef_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': coef
                    }).copy()

                    # Absolute values for importance ranking
                    coef_df['AbsValue'] = coef_df['Value'].abs()
                    coef_df['Value'] = np.exp(coef_df['Value'])
                    coef_df = coef_df.sort_values(by='Value', ascending=False).reset_index(drop=True)

                    top_20_sum = coef_df.head(20)['Value'].sum()
                    total_sum = coef_df['Value'].sum()
                    relative_importance = (top_20_sum / total_sum) * 100
                    st.markdown(f"**Top 20 Features contribute:** `{relative_importance:.2f}%` of total coefficient magnitude")
                    plot_coefficients_paged(coef_df, "🟩 Factors pointing toward (De) escalation", "pos_slider")

                else:
                    st.warning("Model does not contain coefficients.")
            except Exception as e:
                st.error(f"Error extracting coefficients: {e}")
        else:
            st.warning(f"{model_choice} model not loaded.")

    elif model_choice == "Decision Tree":
        if model is not None:
            try:
                st.write("### 🌳 Decision Tree Visualization with Edge Labels")

                # 1) Load dataset
                dataset = load_dataset(triage_level, escalation_type)
                if dataset is None:
                    st.warning("Dataset not available.")
                    st.stop()
                if 'Escalatie' not in dataset.columns:
                    st.error("Dataset missing 'Escalatie'.")
                    st.stop()

                X = dataset.drop('Escalatie', axis=1)
                y = dataset['Escalatie']
                if hasattr(model, "feature_names_in_"):
                    X = X[model.feature_names_in_]
                feature_names = X.columns.tolist()
                class_names = ["no Escalation", "(De) Escalation"]

                from sklearn.tree import _tree
                import graphviz

                # Custom DOT generator (same as before)
                def decision_tree_to_dot_with_edge_labels(model, feature_names, class_names):
                    tree_ = model.tree_
                    dot = ['digraph Tree {']
                    dot.append('node [shape=box, style="filled, rounded", color="black", fontname=helvetica];')
                    dot.append('edge [fontname=helvetica];')

                    def recurse(node, depth):
                        if tree_.feature[node] != _tree.TREE_UNDEFINED:
                            original_name = feature_names[tree_.feature[node]]
                            is_inverted = (
                                original_name.startswith("triageTitle") or
                                original_name.startswith("age") or
                                original_name.startswith("bodyAreaTitle")
                            )
                            name = original_name
                            threshold = tree_.threshold[node]

                            q_answer = ""
                            if "Q_" in original_name:
                                if threshold <= -0.5:
                                    q_answer = "\\nAnswer: no"
                                elif threshold <= 0.5:
                                    q_answer = "\\nAnswer: yes"

                            label = f"{name}{q_answer}\\nsamples = {tree_.n_node_samples[node]}"
                            dot.append(f'{node} [label="{label}", fillcolor="#e5813900"];')

                            left = tree_.children_left[node]
                            right = tree_.children_right[node]

                            recurse(left, depth + 1)
                            recurse(right, depth + 1)

                            if is_inverted:
                                dot.append(f'{node} -> {left} [label="False"];')
                                dot.append(f'{node} -> {right} [label="True"];')
                            else:
                                dot.append(f'{node} -> {left} [label="True"];')
                                dot.append(f'{node} -> {right} [label="False"];')
                        else:
                            value = tree_.value[node][0]
                            class_index = value.argmax()
                            label = f"predicted = {class_names[class_index]}\\nsamples = {tree_.n_node_samples[node]}"
                            dot.append(f'{node} [label="{label}", fillcolor="#e5813980"];')

                    recurse(0, 0)
                    dot.append("}")
                    return "\n".join(dot)

                # Display using SVG (high-res)
                with st.expander("🧩 Full Decision Tree (High-Res SVG)"):
                    dot_data = decision_tree_to_dot_with_edge_labels(model, feature_names, class_names)
                    graph = graphviz.Source(dot_data, format="svg")

                    # Render and embed SVG
                    svg = graph.pipe().decode("utf-8")
                    st.components.v1.html(svg, height=800, scrolling=True)
                # 4) Local explanation
                st.write("### 🔍 Local Explanation for a Selected Instance")
                selected_index = st.selectbox("Choose instance index", X.index.tolist())
                x_instance = X.loc[selected_index].values.reshape(1, -1)

                # Path and node tracking
                tree_ = model.tree_
                node_indicator = model.decision_path(x_instance)
                path_nodes = set(node_indicator.indices)
                leaf_id = model.apply(x_instance)[0]

                def local_path_dot(model, x_instance, feature_names, class_names, path_nodes):
                    tree_ = model.tree_
                    dot = ['digraph Tree {',
                        'node [shape=box, style="filled, rounded", fontname=helvetica];',
                        'edge [fontname=helvetica];']

                    for node in path_nodes:
                        if tree_.feature[node] != _tree.TREE_UNDEFINED:
                            original_name = feature_names[tree_.feature[node]]
                            is_inverted = (
                                original_name.startswith("triageTitle") or
                                original_name.startswith("age") or
                                original_name.startswith("bodyAreaTitle")
                            )
                            name = original_name
                            threshold = tree_.threshold[node]
                            value = x_instance[0, tree_.feature[node]]
                            decision = "<=" if value <= threshold else ">"
                            q_answer = ""

                            if "Q_" in original_name:
                                if threshold <= -0.5:
                                    q_answer = "\\nAnswer: no"
                                elif threshold <= 0.5:
                                    q_answer = "\\nAnswer: yes"

                            label = f"{name}{q_answer}\\n({value:.2f} {decision} {threshold:.2f})"
                            dot.append(f'{node} [label="{label}", fillcolor="#a0c4ff"];')
                        else:
                            value = tree_.value[node][0]
                            class_index = value.argmax()
                            label = f"predicted = {class_names[class_index]}\\nsamples = {tree_.n_node_samples[node]}"
                            dot.append(f'{node} [label="{label}", fillcolor="#ffb3c1"];')

                    # Edges in path
                    children_left = tree_.children_left
                    children_right = tree_.children_right
                    for node in path_nodes:
                        left, right = children_left[node], children_right[node]
                        if left in path_nodes:
                            dot.append(f'{node} -> {left};')
                        if right in path_nodes:
                            dot.append(f'{node} -> {right};')

                    dot.append('}')
                    return '\n'.join(dot)

                local_dot = local_path_dot(model, x_instance, feature_names, class_names, path_nodes)
                st.graphviz_chart(local_dot)

                # Get predicted class
                leaf_node = model.apply(x_instance)[0]
                predicted_class = class_names[tree_.value[leaf_node][0].argmax()]

                # Get true class
                true_class = y.loc[selected_index]

                # Build explanation text
                explanation_lines = []
                for node in sorted(path_nodes):
                    if tree_.feature[node] != _tree.TREE_UNDEFINED:
                        feature = feature_names[tree_.feature[node]]
                        value = x_instance[0, tree_.feature[node]]
                        threshold = tree_.threshold[node]
                        decision = "no" if value <= threshold else "yes"
                        explanation_lines.append(f"{feature}: {decision}")

                # Display explanation summary
                st.markdown(f"**True class: {class_names[true_class]}**")
                st.markdown(f"**Predicted class: {predicted_class}**")
                st.markdown(f"**The patient was predicted to have been _{predicted_class}_ because of:**")
                st.markdown(" • " + "<br> • ".join(explanation_lines), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error plotting Decision Tree with Graphviz: {e}")
        else:
            st.warning("Decision Tree model not loaded.")




    else:
        st.info("Global explanations are only available for EBM, Logistic Regression, SVM, and Decision Trees.")

    # --------------------------
    # 🔬 Local Explanation (EBM)
    # --------------------------
    st.markdown("---")
    st.write("### 🔬 Local Explanation")

    if model_choice != "Explainable Boosting Machine (EBM)":
        st.info("Please select EBMs for local explainations here.")
    elif model is not None:
        dataset = load_dataset(triage_level, escalation_type)
        if dataset is not None:
            if 'quality' in dataset.columns:
                X = dataset.drop('quality', axis=1)
                y = dataset['quality']
            elif 'Escalatie' in dataset.columns:
                X = dataset.drop('Escalatie', axis=1)
                y = dataset['Escalatie']
            else:
                X = dataset
                y = pd.Series([None] * len(X))

            total_instances = X.shape[0]
            st.write(f"Total instances: **{total_instances}**")

            row_index = st.number_input("Enter a row index to explain", min_value=0, max_value=total_instances - 1, value=0, step=1)
            row = X.iloc[[row_index]]
            pred = model.predict(row)[0]
            true_label = y.iloc[row_index] if row_index < len(y) else "N/A"

            st.write(f"**Prediction:** `{pred:.3f}`")
            st.write(f"**True Label:** `{true_label}`")

            local_exp = model.explain_local(row)
            local_data = local_exp.data(0)
            local_data['names'] = [f"{name} = {value}" for name, value in zip(local_data["names"], local_data["values"])]
            # Create contributions Series
            contributions = pd.Series(local_data["scores"], index=local_data["names"])

            # Sort by absolute value
            sorted_contributions = contributions.abs().sort_values(ascending=False)
            import altair as alt

            # Prepare DataFrame
            top_contributions = contributions.abs().sort_values(ascending=False).head(20)
            feature_names = top_contributions.index
            values = contributions.loc[feature_names]  # get original signed values

            df_plot = pd.DataFrame({
                "Feature": feature_names,
                "Contribution": values.values
            })

            # Create Altair bar chart
            chart = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X('Contribution:Q'),
                y=alt.Y('Feature:N', sort='-x'),  # this keeps the features sorted by contribution
                color=alt.condition(
                    alt.datum.Contribution > 0,
                    alt.value("steelblue"),
                    alt.value("tomato")
                ),
                tooltip=["Feature", "Contribution"]
            ).properties(
                height=500,
                title="Top 20 Feature Contributions (by absolute value)"
            )

            st.altair_chart(chart, use_container_width=True)


            bias = local_data["extra"].get("bias", None)
            if bias is not None:
                st.write(f"**Intercept (bias):** `{bias:.3f}`")
        else:
            st.warning("Local dataset not available.")
