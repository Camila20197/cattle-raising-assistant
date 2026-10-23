"""Chat interface for producers (and for live demos) on top of the FastAPI service.

The UI only talks to the API over HTTP (``API_URL``): it knows nothing about
retrieval, prompts or Groq, so either side can change without touching the
other (low coupling, message-style communication between components).

Design follows Nielsen's 10 usability heuristics; the numbers in the
comments (H1..H10) point to the heuristic each element addresses.

Run locally (with the API running on :8000):
    uv run python -m streamlit run src/ui/chat_app.py --server.port 8502
"""

import os
import re

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 90
MAX_QUESTION_LENGTH = 500
# Soft per-visitor limit (a reload starts a new session). The hard cost
# ceiling is the API's global quota (MAX_QUESTIONS_PER_MINUTE / _PER_DAY).
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "15"))

# H2 - Speak the user's language: show report titles, not file names.
DOCUMENT_TITLES = {
    "INTA_CICVyA_InstitutodeVirologia_Maidana_SS_Efecto_antiviral_de_la_ivermectina.pdf":
        "Efecto antiviral de la ivermectina sobre el alfaherpesvirus bovino 1 (INTA, 2025)",
    "INTA_CICVyA_InstitutodeVirologia_Romera_SA_Primer_registro_de_alfaherpesvirus_bovino.pdf":
        "Primer registro de alfaherpesvirus bovino 1.1 en garrapatas R. microplus (INTA, 2025)",
    "INTA_CREntreRíos_EEAConcepcióndelUruguay_GorelikZonis_YG_Impacto_sequía_ganadería_bovina_Gualeguaychu.pdf":
        "Impacto de la sequía en la ganadería bovina de Gualeguaychú (INTA, 2026)",
    "INTA_CRSantaFe_EEARafaela_Nava_S_Guia_identificacion_especies_garrapatas_Entre_Rios.pdf":
        "Guía para identificar garrapatas de bovinos en Entre Ríos (INTA)",
    "INTA_CRBsAs_EEABalcarce_Canton_GJ_Presentaci_n_casos_relevantes_diagn_stico.pdf":
        "Casos relevantes de diagnóstico veterinario en bovinos (INTA Balcarce, 2026)",
    "INTA_CRBsAsSur_EEACuencadelSalado_Maresca_S_Indices_reproductivos_rodeos_de_cria.pdf":
        "Índices reproductivos en rodeos de cría (INTA, 2026)",
    "INTA_CRLaPampa-SanLuis_EEAAnguil_Rojas_MDC_Suplementaci_n_microelementos_rendimiento_reproductivo_servicio.pdf":
        "Suplementación con microelementos y rendimiento reproductivo (INTA, 2026)",
}

# H6 - Recognition rather than recall: ready-made questions to click.
EXAMPLE_QUESTIONS = [
    "¿Qué especies de garrapatas afectan a los bovinos en Entre Ríos?",
    "¿Cómo afectó la sequía a la ganadería de Gualeguaychú?",
    "¿Qué enfermedades causa el alfaherpesvirus bovino tipo 1?",
    "¿Tiene la ivermectina efecto antiviral en bovinos?",
    "¿Qué plantas tóxicas causaron intoxicaciones en bovinos?",
    "¿Cuáles son los índices de preñez y destete de referencia en rodeos de cría?",
]

WELCOME_MESSAGE = (
    "¡Hola! Soy un asistente que responde preguntas sobre **sanidad bovina** "
    "usando informes técnicos del **INTA**. Cada respuesta indica de qué "
    "documento y página sale, para que puedas verificarla."
)


ELLIPSIS = re.compile(r"…|\.\.\.")
# Models write citations as [file.pdf: 3] or 【file.pdf: 3】, sometimes with the
# file name shortened (e.g. "INTA…_Guia_garrapatas.pdf").
CITATION = re.compile(r"[\[【]\s*([^\[\]【】]+?\.pdf)\s*[:：]\s*(\d+)\s*[\]】]")


def resolve_document(file_name: str) -> str | None:
    """Find the full file name for an exact or abbreviated ("…") citation."""
    if file_name in DOCUMENT_TITLES:
        return file_name
    parts = [part for part in ELLIPSIS.split(file_name) if part]
    if len(parts) < 2 and not ELLIPSIS.search(file_name):
        return None
    head, tail = (parts[0], parts[-1]) if parts else ("", "")
    matches = [name for name in DOCUMENT_TITLES if name.startswith(head) and name.endswith(tail)]
    return matches[0] if len(matches) == 1 else None


def document_title(file_name: str) -> str:
    resolved = resolve_document(file_name)
    if resolved:
        return DOCUMENT_TITLES[resolved]
    readable = re.sub(r"[_-]+", " ", ELLIPSIS.sub(" ", file_name.removesuffix(".pdf")))
    return re.sub(r"\s+", " ", readable).strip()


def friendly_citations(answer: str) -> str:
    """Replace ``[file.pdf: 3]`` / ``【file.pdf: 3】`` citations with readable ones (H2, H8)."""

    def replace(match: re.Match) -> str:
        title = document_title(match.group(1).strip()).split(" (")[0]
        return f"*[{title}, pág. {match.group(2)}]*"

    return CITATION.sub(replace, answer)


def check_api() -> dict | None:
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def ask_api(question: str) -> dict:
    """Call POST /ask and turn every failure into a message a producer understands (H9)."""
    try:
        response = httpx.post(
            f"{API_URL}/ask", json={"question": question}, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except httpx.TimeoutException:
        return {"error": "La respuesta está tardando demasiado. Probá de nuevo en un momento."}
    except httpx.HTTPError:
        return {
            "error": "No me pude conectar con el servicio del asistente. "
            "Verificá que la API esté encendida e intentá otra vez."
        }

    if response.status_code == 429:
        return {
            "error": "El asistente recibió muchas preguntas en poco tiempo. "
            "Esperá un momento y volvé a intentar."
        }
    if response.status_code == 503:
        return {
            "error": "El modelo de lenguaje no está disponible en este momento. "
            "Probá de nuevo en unos minutos.",
            "detail": response.json().get("detail", ""),
        }
    if response.status_code >= 400:
        return {"error": f"Ocurrió un error inesperado (código {response.status_code})."}
    return response.json()


def send_feedback(conversation_id: int, rating: str) -> bool:
    try:
        response = httpx.post(
            f"{API_URL}/feedback",
            json={"conversation_id": conversation_id, "rating": rating},
            timeout=10,
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Fuentes consultadas ({len(sources)})"):
        for source in sources:
            st.markdown(f"- {document_title(source['document'])} — página {source['page']}")


def render_feedback(message_index: int, conversation_id: int) -> None:
    """Thumbs up/down under each answer (H1: confirm the vote was saved)."""
    key = f"feedback_{message_index}"
    selection = st.feedback("thumbs", key=key)
    sent_key = f"{key}_sent"
    if selection is not None and not st.session_state.get(sent_key):
        rating = "up" if selection == 1 else "down"
        if send_feedback(conversation_id, rating):
            st.session_state[sent_key] = True
            st.toast("¡Gracias! Tu opinión nos ayuda a mejorar.")
        else:
            st.toast("No se pudo guardar tu opinión.")


def render_message(index: int, message: dict) -> None:
    with st.chat_message(message["role"], avatar="🐄" if message["role"] == "assistant" else "🧑‍🌾"):
        if message.get("error"):
            st.warning(message["error"])
            if message.get("detail"):
                st.caption(message["detail"])
            return
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            if message.get("conversation_id"):
                render_feedback(index, message["conversation_id"])


def handle_question(question: str) -> None:
    question = question.strip()[:MAX_QUESTION_LENGTH]
    if not question or session_limit_reached():
        return
    st.session_state.questions_asked += 1
    st.session_state.messages.append({"role": "user", "content": question})
    # H1 - Visibility of system status while the answer is generated.
    with st.spinner("Buscando en los informes del INTA y redactando la respuesta…"):
        result = ask_api(question)

    if "error" in result:
        st.session_state.messages.append(
            {"role": "assistant", "error": result["error"], "detail": result.get("detail")}
        )
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": friendly_citations(result["answer"]),
                "sources": result.get("sources", []),
                "conversation_id": result.get("conversation_id"),
            }
        )


def session_limit_reached() -> bool:
    return bool(MAX_QUESTIONS_PER_SESSION) and (
        st.session_state.questions_asked >= MAX_QUESTIONS_PER_SESSION
    )


def render_sidebar(health: dict | None) -> None:
    with st.sidebar:
        st.header("🐄 Asistente ganadero")
        # H1 - System status.
        if health:
            st.success("Servicio en línea")
            st.caption(f"Modelo: `{health.get('generation_model', '?')}`")
        else:
            st.error("El servicio no responde")
            st.caption(f"No se pudo contactar a `{API_URL}`.")

        # H3 - User control and freedom: a clear way to start over.
        if st.button("🔄 Nueva conversación", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # H10 - Help and documentation.
        with st.expander("¿Cómo funciona?"):
            st.markdown(
                "1. Escribís una pregunta.\n"
                "2. El asistente busca los fragmentos más relevantes en los informes del INTA.\n"
                "3. Un modelo de lenguaje redacta la respuesta **usando solo esos fragmentos** "
                "y cita documento y página.\n"
                "4. Si los informes no tienen la información, te lo dice en lugar de inventar."
            )
        with st.expander("Documentos disponibles"):
            for title in DOCUMENT_TITLES.values():
                st.markdown(f"- {title}")

        st.divider()
        st.caption(
            "⚠️ Información de referencia: no reemplaza la consulta con un/a "
            "médico/a veterinario/a. El asistente no diagnostica ni indica dosis."
        )


def main() -> None:
    st.set_page_config(page_title="Asistente ganadero — INTA", page_icon="🐄", layout="centered")
    st.session_state.setdefault("messages", [])
    # Not reset by "Nueva conversación", so the button can't bypass the limit.
    st.session_state.setdefault("questions_asked", 0)

    health = check_api()
    render_sidebar(health)

    st.title("Asistente de sanidad bovina")
    st.caption("Respuestas basadas en informes técnicos del INTA, con fuentes citadas.")

    if not st.session_state.messages:
        st.info(WELCOME_MESSAGE)
        st.markdown("**Probá con una de estas preguntas:**")
        columns = st.columns(2)
        for position, example in enumerate(EXAMPLE_QUESTIONS):
            if columns[position % 2].button(example, key=f"example_{position}", use_container_width=True):
                handle_question(example)
                st.rerun()

    for index, message in enumerate(st.session_state.messages):
        render_message(index, message)

    # H5 - Error prevention: input disabled while the service is down, length limit.
    limit_reached = session_limit_reached()
    if limit_reached:
        st.info(
            f"Llegaste al máximo de {MAX_QUESTIONS_PER_SESSION} preguntas de esta demo. "
            "¡Gracias por probar el asistente!"
        )
    question = st.chat_input(
        "Escribí tu pregunta sobre sanidad bovina…",
        max_chars=MAX_QUESTION_LENGTH,
        disabled=health is None or limit_reached,
    )
    if question:
        handle_question(question)
        st.rerun()


if __name__ == "__main__":
    main()
