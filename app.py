import streamlit as st

from src.config import SUPPORTED_SPORTS, DIFFICULTY_LEVELS, validate_config
from src.database import setup_and_populate_db
from src.generator import compile_quiz_data


# ---------------------------------------------------------------------------
# 1. One-time setup: seed the vector DB with offline facts
# ---------------------------------------------------------------------------
@st.cache_resource
def prepare_knowledge_base():
    return setup_and_populate_db()


prepare_knowledge_base()

st.set_page_config(page_title="Sports Quiz Agent", page_icon="🏆", layout="centered")

st.title("🏆 AI-Powered Sports Quiz Generator")
st.write(
    "Generates fresh, factually-grounded multiple-choice sports quizzes using "
    "**RAG** — a local ChromaDB knowledge base combined with live DuckDuckGo web search."
)

config_warnings = validate_config()
for w in config_warnings:
    st.warning(w)

# ---------------------------------------------------------------------------
# 2. Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Quiz Settings")
sport_choice = st.sidebar.selectbox("Select Sport", SUPPORTED_SPORTS)
difficulty = st.sidebar.select_slider("Select Difficulty", options=DIFFICULTY_LEVELS)
num_questions = st.sidebar.slider("Number of Questions", min_value=3, max_value=5, value=4)

# ---------------------------------------------------------------------------
# 3. Session state
# ---------------------------------------------------------------------------
if "questions" not in st.session_state:
    st.session_state.questions = None
    st.session_state.raw_text = None
    st.session_state.context_used = None
    st.session_state.answers = {}

if st.sidebar.button("🎲 Generate Fresh Quiz", use_container_width=True):
    with st.spinner("Retrieving historic facts and scouring the live web..."):
        try:
            questions, raw_text, context_used = compile_quiz_data(
                sport_choice, difficulty, num_questions
            )
            if not questions:
                st.error(
                    "The model's response couldn't be parsed into questions. "
                    "Try regenerating — this can happen occasionally with free-form output."
                )
            st.session_state.questions = questions
            st.session_state.raw_text = raw_text
            st.session_state.context_used = context_used
            st.session_state.answers = {}
        except Exception as e:
            st.error(f"Failed to generate quiz: {e}")

# ---------------------------------------------------------------------------
# 4. Render the quiz
# ---------------------------------------------------------------------------
if st.session_state.questions:
    st.subheader(f"Quiz: {sport_choice} · {difficulty}")

    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i + 1}. {q['question']}**")

        selected = st.radio(
            label=f"question_{i}",
            options=list(q["options"].keys()),
            format_func=lambda k, q=q: f"{k}) {q['options'][k]}",
            key=f"radio_{i}",
            index=None,
            label_visibility="collapsed",
        )

        if selected:
            st.session_state.answers[i] = selected
            if selected == q["correct"]:
                st.success(f"✅ Correct! Answer: {q['correct']}) {q['options'][q['correct']]}")
            else:
                st.error(
                    f"❌ Not quite. Correct answer: {q['correct']}) {q['options'][q['correct']]}"
                )
            with st.container(border=True):
                st.caption("Explanation")
                st.write(q["explanation"])

        st.divider()

    score = sum(
        1 for i, q in enumerate(st.session_state.questions)
        if st.session_state.answers.get(i) == q["correct"]
    )
    answered = len(st.session_state.answers)
    if answered:
        st.info(f"Score so far: {score} / {answered} answered")

    with st.expander("📋 Raw quiz text (copy-paste for social media)"):
        st.text_area("Raw output", value=st.session_state.raw_text, height=250, label_visibility="collapsed")

    with st.expander("🔍 Inspect Ground Truth (RAG context used)"):
        st.code(st.session_state.context_used, language="markdown")
else:
    st.info("👈 Choose a sport and difficulty, then click **Generate Fresh Quiz** to begin.")
