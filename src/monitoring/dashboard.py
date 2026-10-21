from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.monitoring.db import get_connection

load_dotenv()

st.set_page_config(page_title="Cattle Assistant - Monitoring", layout="wide")


def _query_df(sql: str) -> pd.DataFrame:
    """Run a query and return the results as a DataFrame, without relying on
    pandas' SQLAlchemy-only fast path (psycopg3 connections work fine as a
    plain DBAPI connection, this just avoids a compatibility warning)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


@st.cache_data(ttl=30)
def load_conversations() -> pd.DataFrame:
    return _query_df(
        "SELECT id, created_at, question, answer, retrieval_method, prompt_variant, "
        "COALESCE(generation_model, 'unknown') AS generation_model "
        "FROM conversations ORDER BY created_at"
    )


@st.cache_data(ttl=30)
def load_feedback() -> pd.DataFrame:
    return _query_df(
        "SELECT id, conversation_id, created_at, rating, comment "
        "FROM feedback ORDER BY created_at"
    )


st.title("🐄 Cattle Raising Assistant — Monitoring")

conversations = load_conversations()
feedback = load_feedback()

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Conversaciones totales", len(conversations))
col2.metric("Feedback recibido", len(feedback))
positive_pct = (feedback["rating"].eq("up").mean() * 100) if len(feedback) else 0
col3.metric("Feedback positivo", f"{positive_pct:.0f}%")

if conversations.empty:
    st.info(
        "Todavía no hay conversaciones registradas. Hacé algunas preguntas "
        "en /ask y volvé a cargar esta página."
    )
    st.stop()

conversations["date"] = pd.to_datetime(conversations["created_at"]).dt.date
conversations["answer_length"] = conversations["answer"].str.len()

# Paleta campo: verdes y amarillos, alternados para que cada gráfico se
# distinga del anterior.
GREEN_DARK = "#2E7D32"
GREEN = "#4C9A2A"
GREEN_LIGHT = "#7CB342"
YELLOW = "#F9A825"
YELLOW_LIGHT = "#FBC02D"

# --- Chart 1 ---
st.subheader("Conversaciones por día")
st.bar_chart(conversations.groupby("date").size(), color=GREEN)

# --- Chart 2 ---
st.subheader("Feedback: positivo vs. negativo")
if not feedback.empty:
    st.bar_chart(feedback["rating"].value_counts(), color=YELLOW)
else:
    st.caption("Todavía no se recibió feedback.")

# --- Chart 3 ---
st.subheader("Método de retrieval usado")
st.bar_chart(conversations["retrieval_method"].value_counts(), color=GREEN_LIGHT)

# --- Chart 4 ---
st.subheader("Variante de prompt usada")
st.bar_chart(conversations["prompt_variant"].value_counts(), color=YELLOW_LIGHT)

# --- Chart 5 ---
st.subheader("Modelo de lenguaje usado")
st.bar_chart(conversations["generation_model"].value_counts(), color=GREEN)

# --- Chart 6 ---
st.subheader("Longitud promedio de respuesta por día (caracteres)")
st.line_chart(conversations.groupby("date")["answer_length"].mean(), color=GREEN_DARK)

# --- Chart 7 ---
st.subheader("Preguntas más frecuentes")
st.bar_chart(conversations["question"].value_counts().head(10), color=YELLOW)

st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")