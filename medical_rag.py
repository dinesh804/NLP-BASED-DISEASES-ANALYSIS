import re
import pandas as pd
import streamlit as st

from symptom import analyze_query

# Keep heavy RAG/model initialization inside the button flow.
from app import (
    VECTORSTORE_PATH,
    create_embeddings,
    create_vectorstore,
    load_medlineplus,
    prepare_documents,
    create_chunks,
    load_vectorstore,
    create_retriever,
    create_llm,
    create_prompt,
    build_enhanced_query,
    retrieve_context,
    format_context,
    generate_answer,
)


st.set_page_config(
    page_title="Medical RAG Analysis Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        background: #ffffff;
        min-height: 105px;
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.88rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }
    .answer-box {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        background: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🩺 Medical RAG Analysis Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">NLP symptom extraction + FAISS retrieval + MedlinePlus + Ollama Llama 3.2</div>',
    unsafe_allow_html=True,
)


# -----------------------------
# Session state
# -----------------------------
defaults = {
    "rag_ready": False,
    "retriever": None,
    "llm": None,
    "prompt": None,
    "history": [],
    "last_analysis": None,
    "last_documents": [],
    "last_answer": "",
    "last_question": "",
    "last_profile": {},
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Helpers
# -----------------------------
def symptom_names(analysis):
    values = []
    for item in analysis.get("present_symptoms", []):
        if isinstance(item, dict):
            name = item.get("symptom", "")
        else:
            name = str(item)
        if name:
            values.append(name)
    return values


def absent_names(analysis):
    values = []
    for item in analysis.get("absent_symptoms", []):
        if isinstance(item, dict):
            name = item.get("symptom", "")
        else:
            name = str(item)
        if name:
            values.append(name)
    return values


def get_condition_lines(answer):
    """
    Display-only extraction from the LLM response.
    These are possible conditions mentioned by the model,
    not clinical probabilities.
    """
    lines = []
    in_section = False

    for raw in answer.splitlines():
        line = raw.strip()

        if re.match(r"^2[\.\)]?\s*POSSIBLE CONDITIONS", line, re.I):
            in_section = True
            continue

        if in_section and re.match(r"^(3[\.\)]?\s*WHAT TO DO|WHAT TO DO)", line, re.I):
            break

        if in_section and line:
            clean = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
            if clean:
                lines.append(clean)

    return lines[:3]


def initialize_rag():
    """Load the existing FAISS database or create it once."""
    progress = []

    progress.append("Loading embedding model")
    embeddings = create_embeddings()

    if VECTORSTORE_PATH.exists():
        progress.append("Loading existing FAISS vector database")
        vectorstore = load_vectorstore(embeddings)
    else:
        progress.append("Creating FAISS vector database")
        raw = load_medlineplus()
        docs = prepare_documents(raw)
        chunks = create_chunks(docs)
        vectorstore = create_vectorstore(chunks, embeddings)

    progress.append("Creating retriever")
    retriever = create_retriever(vectorstore)

    progress.append("Connecting to Ollama / Llama 3.2")
    llm = create_llm()

    progress.append("Creating RAG prompt")
    prompt = create_prompt()

    return retriever, llm, prompt, progress


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Patient Profile")

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=23,
        step=1,
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other", "Prefer not to say"],
    )

    st.divider()

    st.subheader("🔧 System")
    st.write("Knowledge base: **MedlinePlus**")
    st.write("Vector DB: **FAISS**")
    st.write("Embeddings: **all-MiniLM-L6-v2**")
    st.write("LLM: **Ollama / Llama 3.2**")
    st.write("NLP: **spaCy + symptom extraction**")

    st.divider()

    if st.button("🔄 Reset System", use_container_width=True):
        for key in [
            "rag_ready", "retriever", "llm", "prompt",
            "history", "last_analysis", "last_documents",
            "last_answer", "last_question", "last_profile",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


# -----------------------------
# RAG initialization
# -----------------------------
if not st.session_state.rag_ready:
    st.info(
        "Initialize the RAG system once. If the FAISS database already exists, "
        "it will be loaded instead of rebuilt."
    )

    if st.button("🚀 Initialize RAG System", type="primary"):
        try:
            with st.status("Initializing RAG system...", expanded=True) as status:
                retriever, llm, prompt, progress = initialize_rag()

                for message in progress:
                    st.write("✅", message)

                st.session_state.retriever = retriever
                st.session_state.llm = llm
                st.session_state.prompt = prompt
                st.session_state.rag_ready = True

                status.update(
                    label="RAG system ready",
                    state="complete",
                    expanded=False,
                )

            st.rerun()

        except Exception as error:
            st.error("RAG initialization failed.")
            st.exception(error)
            st.stop()
else:
    st.success("🟢 RAG system is ready.")


# -----------------------------
# Dashboard
# -----------------------------
st.divider()
st.header("📊 Analysis Dashboard")

analysis = st.session_state.last_analysis or {}
documents = st.session_state.last_documents or []
answer = st.session_state.last_answer or ""

present = symptom_names(analysis)
absent = absent_names(analysis)
durations = analysis.get("duration", [])
conditions = get_condition_lines(answer)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Symptoms Detected</div>'
        f'<div class="metric-value">{len(present)}</div></div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Possible Conditions</div>'
        f'<div class="metric-value">{len(conditions)}</div></div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Retrieved Topics</div>'
        f'<div class="metric-value">{len(documents)}</div></div>',
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Patient Age</div>'
        f'<div class="metric-value">{age}</div></div>',
        unsafe_allow_html=True,
    )

if not st.session_state.last_analysis:
    st.info("Run an analysis below to populate the dashboard.")


# -----------------------------
# Main input
# -----------------------------
st.divider()
st.header("🔍 Symptom Analysis")

question = st.text_area(
    "Describe the symptoms",
    placeholder="Example: I have fever and stomach pain for 2 days",
    height=130,
)

analyze = st.button(
    "🔍 Analyze Symptoms",
    type="primary",
    use_container_width=True,
)


if analyze:
    if not st.session_state.rag_ready:
        st.warning("Initialize the RAG system first.")
        st.stop()

    if not question.strip():
        st.warning("Please describe the symptoms first.")
        st.stop()

    question = question.strip()

    profile = {
        "age": int(age),
        "gender": gender,
    }

    # NLP
    with st.spinner("Running NLP symptom extraction..."):
        try:
            analysis = analyze_query(question)
        except Exception as error:
            st.error("NLP analysis failed.")
            st.exception(error)
            st.stop()

    # Conservative fallback for common phrases.
    if not analysis.get("present_symptoms"):
        phrases = [
            ("stomach pain", ["stomach pain", "stomach ache"]),
            ("abdominal pain", ["abdominal pain", "belly pain"]),
            ("fever", ["fever"]),
            ("headache", ["headache"]),
            ("vomiting", ["vomiting", "vomit", "throwing up"]),
            ("nausea", ["nausea"]),
            ("diarrhea", ["diarrhea", "diarrhoea"]),
            ("cough", ["cough"]),
            ("cold", ["cold"]),
            ("sore throat", ["sore throat"]),
            ("chest pain", ["chest pain"]),
            ("fatigue", ["fatigue", "tiredness"]),
            ("dizziness", ["dizziness", "dizzy"]),
        ]

        q = question.lower()
        fallback = []
        for canonical, alternatives in phrases:
            if any(p in q for p in alternatives):
                fallback.append({
                    "symptom": canonical,
                    "status": "present",
                    "severity": None,
                })

        if fallback:
            analysis["present_symptoms"] = fallback

    # Retrieval
    with st.spinner("Retrieving MedlinePlus evidence..."):
        enhanced_query = build_enhanced_query(
            question,
            analysis,
            profile,
        )

        try:
            documents = retrieve_context(
                st.session_state.retriever,
                enhanced_query,
            )
        except Exception as error:
            st.error("Medical information retrieval failed.")
            st.exception(error)
            st.stop()

    # Generation
    context = format_context(documents)

    with st.spinner("Generating evidence-grounded response with Llama 3.2..."):
        try:
            answer = generate_answer(
                st.session_state.llm,
                st.session_state.prompt,
                analysis,
                profile,
                context,
                question,
            )
        except Exception as error:
            st.error("RAG response generation failed.")
            st.exception(error)
            st.stop()

    # Save session result
    st.session_state.last_analysis = analysis
    st.session_state.last_documents = documents
    st.session_state.last_answer = answer
    st.session_state.last_question = question
    st.session_state.last_profile = profile

    st.session_state.history.append({
        "Question": question,
        "Symptoms": ", ".join(symptom_names(analysis)),
        "Retrieved Topics": len(documents),
    })
    st.session_state.history = st.session_state.history[-10:]

    st.rerun()


# -----------------------------
# Latest analysis details
# -----------------------------
if st.session_state.last_analysis:
    analysis = st.session_state.last_analysis
    documents = st.session_state.last_documents
    answer = st.session_state.last_answer

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🧠 NLP Analysis",
            "🧬 Possible Conditions",
            "📚 Retrieved Evidence",
            "💬 RAG Response",
        ]
    )

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Present Symptoms")
            if present:
                for item in present:
                    st.write(f"• {item}")
            else:
                st.write("None detected.")

        with col2:
            st.subheader("Absent / Negated")
            if absent:
                for item in absent:
                    st.write(f"• {item}")
            else:
                st.write("None mentioned.")

        with col3:
            st.subheader("Duration")
            if durations:
                for item in durations:
                    st.write(f"• {item}")
            else:
                st.write("Not specified.")

        with st.expander("View enhanced retrieval query"):
            st.code(
                build_enhanced_query(
                    st.session_state.last_question,
                    analysis,
                    st.session_state.last_profile,
                ),
                language="text",
            )

    with tab2:
        st.subheader("Possible Conditions")

        if conditions:
            condition_df = pd.DataFrame({
                "Possible condition": conditions,
                "Interpretation": [
                    "Mentioned by the RAG response based on retrieved MedlinePlus context."
                    for _ in conditions
                ],
            })
            st.dataframe(
                condition_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No possible conditions were extracted from the generated response."
            )

        st.warning(
            "These are evidence-relevant possibilities from the RAG response, "
            "not clinical probability scores or confirmed diagnoses."
        )

    with tab3:
        st.subheader("Top Retrieved MedlinePlus Topics")

        if documents:
            topic_rows = []
            for i, document in enumerate(documents, 1):
                topic_rows.append({
                    "Rank": i,
                    "Topic": document.metadata.get("title", "Unknown"),
                    "Source": document.metadata.get("source", "MedlinePlus"),
                })

            topic_df = pd.DataFrame(topic_rows)
            st.dataframe(
                topic_df,
                use_container_width=True,
                hide_index=True,
            )

            for i, document in enumerate(documents, 1):
                title = document.metadata.get("title", "Unknown")
                url = document.metadata.get("url", "")

                with st.expander(f"{i}. {title}"):
                    if url:
                        st.markdown(f"**MedlinePlus:** {url}")
                    st.write(document.page_content)
        else:
            st.info("No retrieved documents.")

    with tab4:
        st.subheader("Evidence-Grounded Response")
        st.markdown(
            '<div class="answer-box">',
            unsafe_allow_html=True,
        )
        st.markdown(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Patient Profile"):
            st.json(st.session_state.last_profile)


# -----------------------------
# Session history
# -----------------------------
if st.session_state.history:
    st.divider()
    st.header("📈 Session Analysis History")

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )


st.divider()
st.caption(
    "MedlinePlus + NLP + LangChain + FAISS + Ollama | "
    "Educational decision-support prototype. Not a confirmed diagnosis."
)
