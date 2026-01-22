import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def load_eval_data(filename="eval_results.csv"):
    """Load evaluation results into a pandas DataFrame."""
    try:
        return pd.read_csv(filename)
    except Exception as e:
        st.warning(f"Failed to load eval_results.csv: {e}")
        return None

def load_llm_metrics(filename="llm_eval_metrics.csv"):
    """Load LLM metrics into a pandas DataFrame."""
    try:
        return pd.read_csv(filename)
    except Exception as e:
        st.warning(f"Failed to load LLM metrics: {e}")
        return None

def basic_metrics(df):
    """Show basic statistics of evaluation results."""
    total = len(df)
    faithful = df["faithful"].sum() if "faithful" in df else 0
    halluc = total - faithful
    avg_len = df["answer"].str.len().mean() if "answer" in df else 0
    avg_sources = df["num_sources"].mean() if "num_sources" in df else 0

    st.markdown(f"**Total Q&A:** {total}")
    st.markdown(f"**Faithful:** {faithful} ({faithful/total:.1%})")
    st.markdown(f"**Hallucinations:** {halluc} ({halluc/total:.1%})")
    st.markdown(f"**Average answer length:** {avg_len:.1f}")
    st.markdown(f"**Average number of sources:** {avg_sources:.1f}")

def plot_faithful_hist(df):
    """Show pie plot of faithfulness"""
    labels = ["Faithful", "Hallucination"]
    values = [
        df["faithful"].sum() if "faithful" in df else 0,
        len(df) - df["faithful"].sum() if "faithful" in df else 0
    ]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
    st.pyplot(fig)

def show_hallucination_questions(df, n=5):
    """Show top-N questions with hallucinations"""
    if "faithful" not in df:
        return
    st.markdown("**Top 'Hallucination' questions:**")
    df_halluc = df[df["faithful"] == False]
    if df_halluc.empty:
        st.success("No hallucinations detected!")
        return
    st.dataframe(df_halluc[["question", "answer", "llm_feedback"]].head(n))

def plot_answer_length(df):
    """Show histogram of answer lengths"""
    if "answer" not in df:
        return
    lengths = df["answer"].str.len()
    fig, ax = plt.subplots()
    ax.hist(lengths, bins=20)
    ax.set_title("Answer Length Distribution")
    ax.set_xlabel("Answer Length")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def llm_eval_dashboard(filename="llm_eval_metrics.csv"):
    """Show summary of evaluation results"""
    st.subheader("LLM Evaluation Metrics")
    df = load_llm_metrics(filename)
    if df is None or df.empty:
        st.info("No LLM metrics data available.")
        return

    metrics = ["faithfulness", "relevance", "conciseness"]
    means = df[metrics].mean()
    st.markdown("### Average LLM Metrics")
    for m in metrics:
        st.markdown(f"**{m.capitalize()}:** {means[m]:.2f}")

    st.markdown("### Metric Trends")
    for m in metrics:
        fig, ax = plt.subplots()
        ax.plot(df[m], marker="o")
        ax.set_title(f"{m.capitalize()} over time")
        ax.set_ylabel(m.capitalize())
        ax.set_xlabel("Example #")
        st.pyplot(fig)

    st.markdown("### Recent Evaluations")
    st.dataframe(df.tail(5))

def analytics_dashboard(filename="eval_results.csv"):
    """Combined dashboard: run this in your Streamlit app"""
    st.header("Analytics Dashboard")
    df = load_eval_data(filename)
    if df is None or df.empty:
        st.warning("No evaluation data available.")
        return
    basic_metrics(df)
    plot_faithful_hist(df)
    plot_answer_length(df)
    show_hallucination_questions(df)
