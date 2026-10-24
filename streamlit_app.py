import streamlit as st
import json
import re
from datetime import datetime
from html import escape

# ============================================================
# IMPORT YOUR EXISTING PROJECT MODULES
# ============================================================
# Put this file in the SAME folder as:
#   1. rag_pipeline.py   -> your RAG pipeline
#   2. symptom.py        -> your NLP symptom analyzer
#
# Your existing RAG uses:
# MedlinePlus + LangChain + FAISS + HuggingFace Embeddings + Ollama
# Your symptom.py uses spaCy + symptom vocabulary + negation + severity
# ============================================================

from app import (
    create_embeddings,
    load_vectorstore,
    create_vectorstore,
    create_chunks,
    load_medlineplus,
    prepare_documents,
    create_retriever,
    create_llm,
    create_prompt,
    retrieve_context,
    format_context,
    generate_answer,
    VECTORSTORE_PATH,
)

from symptom import analyze_query as nlp_analyze_query


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Medical Report Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
/* ---------- GLOBAL ---------- */
:root {
    --bg: #0b0f14;
    --panel: #11161d;
    --panel-2: #171d26;
    --border: #252d38;
    --text: #f3f4f6;
    --muted: #9aa4b2;
    --accent: #ff4d57;
    --accent-hover: #ff626a;
    --blue: #4da3ff;
}

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* ---------- REMOVE STREAMLIT TOP SPACE / DECORATION ---------- */
[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: #0d1218 !important;
    border-right: 1px solid #202733 !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: #0d1218 !important;
}

[data-testid="stSidebar"] * {
    color: #dce1e8 !important;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border-radius: 9px;
    padding: 7px 9px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #171d26 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #252d38 !important;
}

/* ---------- HEADINGS / TEXT ---------- */
h1, h2, h3, h4, h5, h6,
p, label, span, div {
    color: inherit;
}

.stMarkdown, .stCaption {
    color: var(--text);
}

.stCaption {
    color: var(--muted) !important;
}

/* ---------- HERO ---------- */
.hero {
    background: #11161d;
    border: 1px solid #252d38;
    border-left: 4px solid var(--accent);
    padding: 24px 28px;
    border-radius: 14px;
    color: var(--text);
    margin-bottom: 24px;
    box-shadow: none;
}

.hero h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 750;
    color: #ffffff !important;
}

.hero p {
    margin: 7px 0 0;
    color: #9aa4b2 !important;
    font-size: .92rem;
}

/* ---------- CARDS ---------- */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
    box-shadow: none;
    color: var(--text);
}

.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 12px;
}

.mini-label {
    font-size: .76rem;
    color: var(--muted) !important;
    margin-bottom: 4px;
}

.mini-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff !important;
}

/* ---------- METRICS ---------- */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: none;
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #b8c1cd !important;
}

/* ---------- INPUTS ---------- */
.stTextArea textarea,
.stTextInput input,
.stNumberInput input,
[data-baseweb="select"] > div {
    background: #171d26 !important;
    color: #f3f4f6 !important;
    border: 1px solid #303947 !important;
    border-radius: 9px !important;
}

.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #6f7a88 !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #596678 !important;
    box-shadow: 0 0 0 1px #596678 !important;
}

[data-baseweb="select"] span {
    color: #f3f4f6 !important;
}

[data-baseweb="select"] svg {
    fill: #9aa4b2 !important;
}

/* Number input buttons */
[data-testid="stNumberInput"] button {
    background: #171d26 !important;
    color: #dce1e8 !important;
    border-color: #303947 !important;
}

/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #171d26 !important;
    border: 1px solid #303947 !important;
}

[data-baseweb="menu"] li {
    color: #f3f4f6 !important;
}

[data-baseweb="menu"] li:hover {
    background: #222a35 !important;
}

/* ---------- BUTTONS ---------- */
.stButton > button,
.stDownloadButton > button {
    background: #1a2029 !important;
    color: #f3f4f6 !important;
    border: 1px solid #303947 !important;
    border-radius: 9px !important;
    font-weight: 650;
    min-height: 42px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #222a35 !important;
    border-color: #4a5667 !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover) !important;
}

/* ---------- STATUS / ALERTS ---------- */
[data-testid="stStatusWidget"] {
    background: #11161d !important;
    border: 1px solid #252d38 !important;
}

.stAlert {
    background: #151b23 !important;
    border: 1px solid #2a3340 !important;
    color: #e5e7eb !important;
}

.warning {
    background: #1b1710;
    border-left: 3px solid #f59e0b;
    padding: 12px 15px;
    border-radius: 8px;
    margin: 10px 0;
    color: #f3d39a !important;
}

.safe {
    background: #101b17;
    border-left: 3px solid #22c55e;
    padding: 12px 15px;
    border-radius: 8px;
    margin: 10px 0;
    color: #a7e4c2 !important;
}

/* ---------- SYMPTOMS ---------- */
.symptom {
    display: inline-block;
    background: #172333;
    color: #80bdff !important;
    border: 1px solid #29496b;
    border-radius: 999px;
    padding: 6px 11px;
    margin: 4px 5px 4px 0;
    font-size: .82rem;
    font-weight: 600;
}

.absent {
    display: inline-block;
    background: #171b21;
    color: #8d98a6 !important;
    border: 1px solid #303744;
    border-radius: 999px;
    padding: 6px 11px;
    margin: 4px 5px 4px 0;
    font-size: .82rem;
}

.severity-mild {
    color: #4ade80 !important;
    font-weight: 700;
}

.severity-moderate {
    color: #fbbf24 !important;
    font-weight: 700;
}

.severity-severe {
    color: #ff6b6b !important;
    font-weight: 700;
}

/* ---------- SOURCES ---------- */
.source {
    background: #11161d;
    border: 1px solid #252d38;
    border-radius: 10px;
    padding: 13px 15px;
    margin-bottom: 9px;
    color: #e5e7eb;
}

.source a {
    color: #63adff !important;
    text-decoration: none;
    font-weight: 650;
}

.source a:hover {
    text-decoration: underline;
}

/* ---------- TABLES ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid #252d38 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

[data-testid="stDataFrame"] * {
    color: #e5e7eb !important;
}

/* ---------- EXPANDERS ---------- */
[data-testid="stExpander"] {
    background: #11161d !important;
    border: 1px solid #252d38 !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    color: #e5e7eb !important;
}

/* ---------- CODE ---------- */
.stCodeBlock,
pre {
    background: #0f141a !important;
    border: 1px solid #252d38 !important;
    border-radius: 9px !important;
}

/* ---------- DIVIDERS ---------- */
hr {
    border-color: #252d38 !important;
}

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0b0f14;
}

::-webkit-scrollbar-thumb {
    background: #303947;
    border-radius: 8px;
}

::-webkit-scrollbar-thumb:hover {
    background: #465365;
}

/* ---------- MOBILE ---------- */
@media (max-width: 900px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 20px;
    }

    .hero h1 {
        font-size: 1.55rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "report_history" not in st.session_state:
    st.session_state.report_history = []

if "current_report" not in st.session_state:
    st.session_state.current_report = None


# ============================================================
# LOAD AI SYSTEM ONCE
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_ai_system():
    """
    Loads:
        HuggingFace embeddings
        FAISS vector store
        Retriever
        Ollama
        Prompt
    """

    embeddings = create_embeddings()

    if VECTORSTORE_PATH.exists():
        vectorstore = load_vectorstore(embeddings)
    else:
        raw_documents = load_medlineplus()
        documents = prepare_documents(raw_documents)
        chunks = create_chunks(documents)
        vectorstore = create_vectorstore(chunks, embeddings)

    retriever = create_retriever(vectorstore)
    llm = create_llm()
    prompt = create_prompt()

    return retriever, llm, prompt


# ============================================================
# NORMALIZE NLP OUTPUT
# ============================================================

def normalize_nlp_result(raw):
    """
    symptom.py returns:
        symptoms
        duration
        spacy_entities

    The RAG prompt expects:
        present_symptoms
        absent_symptoms
        duration

    This adapter keeps the original NLP output while creating
    the structure required by the RAG pipeline.
    """

    symptoms = raw.get("symptoms", []) or []

    present = []
    absent = []

    for item in symptoms:
        name = item.get("name", "")
        status = item.get("status", "present")
        severity = item.get("severity")

        normalized = {
            "symptom": name,
            "status": status,
            "severity": severity,
        }

        if status == "absent":
            absent.append(normalized)
        else:
            present.append(normalized)

    return {
        "original_query": raw.get("original_query", ""),
        "cleaned_query": raw.get("cleaned_query", ""),
        "present_symptoms": present,
        "absent_symptoms": absent,
        "duration": raw.get("duration", []) or [],
        "spacy_entities": raw.get("spacy_entities", []) or [],
        "raw_symptoms": symptoms,
    }


# ============================================================
# BUILD ENHANCED QUERY
# ============================================================

def build_query_for_rag(question, analysis, age, gender):
    """
    Uses the same idea as your existing RAG pipeline but keeps
    the Streamlit app independent from console input().
    """

    parts = [
        f"Patient age: {age}",
        f"Patient gender: {gender}",
        f"Original query: {question}",
    ]

    present = analysis.get("present_symptoms", [])
    absent = analysis.get("absent_symptoms", [])
    duration = analysis.get("duration", [])

    present_names = [
        x.get("symptom", "")
        for x in present
        if x.get("symptom")
    ]

    absent_names = [
        x.get("symptom", "")
        for x in absent
        if x.get("symptom")
    ]

    severity_items = []

    for item in present:
        if item.get("symptom") and item.get("severity"):
            severity_items.append(
                f"{item['symptom']}: {item['severity']}"
            )

    if present_names:
        parts.append(
            "Present symptoms: " + ", ".join(present_names)
        )

    if absent_names:
        parts.append(
            "Absent symptoms: " + ", ".join(absent_names)
        )

    if duration:
        parts.append(
            "Duration: " + ", ".join(map(str, duration))
        )

    if severity_items:
        parts.append(
            "Severity: " + ", ".join(severity_items)
        )

    return "\n".join(parts)


# ============================================================
# ANSWER SECTION EXTRACTION
# ============================================================

def extract_section(answer, section_number, next_number=None):
    if not answer:
        return ""

    text = str(answer)

    if next_number:
        pattern = (
            rf"{section_number}\.\s*.*?"
            rf"(.*?)(?=\n\s*{next_number}\.\s*|\Z)"
        )
    else:
        pattern = rf"{section_number}\.\s*.*?(.*)$"

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return ""

    return match.group(1).strip()


def clean_answer(answer):
    """
    Keeps the model's original medical answer.
    Only removes accidental empty whitespace.
    """
    return str(answer).strip() if answer else ""


def answer_to_html(answer):
    return escape(clean_answer(answer)).replace("\n", "<br>")


# ============================================================
# REPORT OBJECT
# ============================================================

def create_report(
    question,
    age,
    gender,
    analysis,
    answer,
    retrieved_documents,
):
    now = datetime.now()

    report_id = now.strftime("MR-%Y%m%d-%H%M%S")

    sources = []

    for doc in retrieved_documents:
        sources.append(
            {
                "title": doc.metadata.get("title", "Unknown"),
                "url": doc.metadata.get("url", ""),
                "source": doc.metadata.get(
                    "source",
                    "MedlinePlus / U.S. National Library of Medicine",
                ),
            }
        )

    return {
        "report_id": report_id,
        "created_at": now.isoformat(timespec="seconds"),
        "patient": {
            "age": age,
            "gender": gender,
        },
        "original_question": question,
        "nlp_analysis": analysis,
        "ai_response": clean_answer(answer),
        "possible_conditions_section": extract_section(
            answer, 2, 3
        ),
        "self_care_section": extract_section(
            answer, 3, 4
        ),
        "medical_help_section": extract_section(
            answer, 4, 5
        ),
        "sources": sources,
    }


# ============================================================
# DOWNLOADABLE REPORT
# ============================================================

def report_json(report):
    return json.dumps(
        report,
        indent=4,
        ensure_ascii=False,
    )


def report_text(report):
    analysis = report["nlp_analysis"]

    present = analysis.get("present_symptoms", [])
    absent = analysis.get("absent_symptoms", [])
    duration = analysis.get("duration", [])

    lines = [
        "AI MEDICAL INFORMATION REPORT",
        "=" * 60,
        f"Report ID: {report['report_id']}",
        f"Created: {report['created_at']}",
        "",
        "PATIENT PROFILE",
        "-" * 30,
        f"Age: {report['patient']['age']}",
        f"Gender: {report['patient']['gender']}",
        "",
        "ORIGINAL QUERY",
        "-" * 30,
        report["original_question"],
        "",
        "SYMPTOM ANALYSIS",
        "-" * 30,
    ]

    lines.append(
        "Present symptoms: " +
        (
            ", ".join(
                x.get("symptom", "")
                for x in present
            )
            if present
            else "None detected"
        )
    )

    lines.append(
        "Absent symptoms: " +
        (
            ", ".join(
                x.get("symptom", "")
                for x in absent
            )
            if absent
            else "None detected"
        )
    )

    lines.append(
        "Duration: " +
        (
            ", ".join(map(str, duration))
            if duration
            else "Not detected"
        )
    )

    lines.extend(
        [
            "",
            "AI MEDICAL INFORMATION RESPONSE",
            "-" * 30,
            report["ai_response"],
            "",
            "IMPORTANT",
            "-" * 30,
            "This report is for medical information and decision support.",
            "It is not a confirmed diagnosis and does not replace advice from a healthcare professional.",
            "",
            "SOURCES",
            "-" * 30,
        ]
    )

    for source in report["sources"]:
        lines.append(
            f"{source['title']} | {source['url']}"
        )

    return "\n".join(lines)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🩺 Medical AI")

    page = st.radio(
        "Navigation",
        [
            "🏠 Medical Dashboard",
            "📋 Medical Report",
            "🧠 NLP Analysis",
            "📚 Retrieved Sources",
            "🕘 Report History",
        ],
    )

    st.divider()

    st.markdown("### System")
    st.caption("NLP symptom analysis")
    st.caption("LangChain RAG")
    st.caption("FAISS retrieval")
    st.caption("MedlinePlus knowledge base")
    st.caption("Ollama Llama 3.2")

    st.divider()

    st.warning(
        "Medical information only. "
        "The system does not provide a confirmed diagnosis."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">
    <h1>🩺 AI Medical Report Dashboard</h1>
    <p>
        NLP Symptom Analysis • MedlinePlus RAG • FAISS • LangChain • Ollama
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1 — MEDICAL DASHBOARD
# ============================================================

if page == "🏠 Medical Dashboard":

    st.markdown("### Patient Consultation")

    left, right = st.columns([2.2, 1])

    with left:
        question = st.text_area(
            "Describe your symptoms",
            height=150,
            placeholder=(
                "Example: I have fever and severe headache "
                "for 3 days."
            ),
        )

    with right:
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=25,
            step=1,
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other",
                "Prefer not to say",
            ],
        )

    analyze_button = st.button(
        "🔍 Analyze & Generate Medical Report",
        type="primary",
        use_container_width=True,
    )

    if analyze_button:

        if not question.strip():
            st.error("Please describe the symptoms first.")
            st.stop()

        try:
            with st.status(
                "Running medical analysis...",
                expanded=True,
            ) as status:

                st.write("Loading NLP symptom analyzer...")
                raw_analysis = nlp_analyze_query(
                    question.strip()
                )

                st.write("Normalizing symptoms, negation and severity...")
                analysis = normalize_nlp_result(
                    raw_analysis
                )

                st.write("Loading RAG system...")
                retriever, llm, prompt = initialize_ai_system()

                st.write("Building patient-aware retrieval query...")
                enhanced_query = build_query_for_rag(
                    question.strip(),
                    analysis,
                    age,
                    gender,
                )

                st.write("Retrieving MedlinePlus information...")
                retrieved_documents = retrieve_context(
                    retriever,
                    enhanced_query,
                )

                context = format_context(
                    retrieved_documents
                )

                st.write("Generating grounded medical information...")
                answer = generate_answer(
                    llm,
                    prompt,
                    analysis,
                    {
                        "age": age,
                        "gender": gender,
                    },
                    context,
                    question.strip(),
                )

                report = create_report(
                    question=question.strip(),
                    age=age,
                    gender=gender,
                    analysis=analysis,
                    answer=answer,
                    retrieved_documents=retrieved_documents,
                )

                st.session_state.current_report = report

                st.session_state.report_history.insert(
                    0,
                    report,
                )

                # Keep last 20 reports in memory.
                st.session_state.report_history = (
                    st.session_state.report_history[:20]
                )

                status.update(
                    label="Medical report generated",
                    state="complete",
                )

        except Exception as error:
            st.error("The analysis could not be completed.")
            st.exception(error)
            st.stop()

    report = st.session_state.current_report

    if report:

        analysis = report["nlp_analysis"]

        st.markdown("---")
        st.markdown("### Report Overview")

        present = analysis.get(
            "present_symptoms", []
        )

        absent = analysis.get(
            "absent_symptoms", []
        )

        duration = analysis.get(
            "duration", []
        )

        severe_count = sum(
            1
            for item in present
            if item.get("severity") == "severe"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Symptoms Detected",
                len(present),
            )

        with c2:
            st.metric(
                "Absent / Negated",
                len(absent),
            )

        with c3:
            st.metric(
                "Severe Symptoms",
                severe_count,
            )

        with c4:
            st.metric(
                "Sources Retrieved",
                len(report["sources"]),
            )

        st.markdown("### Patient Summary")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="mini-label">Age</div>
                    <div class="mini-value">{report['patient']['age']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="mini-label">Gender</div>
                    <div class="mini-value">{escape(str(report['patient']['gender']))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p3:
            st.markdown(
                f"""
                <div class="card">
                    <div class="mini-label">Report ID</div>
                    <div class="mini-value" style="font-size:1.05rem">
                        {report['report_id']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 🧠 Detected Symptoms")

        if present:
            symptom_html = ""

            for item in present:
                name = escape(
                    str(item.get("symptom", "Unknown"))
                )
                severity = item.get("severity")

                if severity:
                    symptom_html += (
                        f'<span class="symptom">'
                        f'{name} · {escape(str(severity))}'
                        f'</span>'
                    )
                else:
                    symptom_html += (
                        f'<span class="symptom">{name}</span>'
                    )

            st.markdown(
                symptom_html,
                unsafe_allow_html=True,
            )
        else:
            st.info("No present symptoms were detected.")

        if absent:
            st.markdown("#### Negated / Absent Symptoms")

            absent_html = ""

            for item in absent:
                absent_html += (
                    f'<span class="absent">'
                    f"{escape(str(item.get('symptom', 'Unknown')))}"
                    f"</span>"
                )

            st.markdown(
                absent_html,
                unsafe_allow_html=True,
            )

        if duration:
            st.markdown(
                "#### Duration"
            )
            st.write(
                " • ".join(
                    str(x)
                    for x in duration
                )
            )

        st.markdown("---")
        st.markdown("### 🤖 AI Medical Information")

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            answer_to_html(
                report["ai_response"]
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="warning">
                <b>Important:</b> This is medical information and
                decision support generated from the retrieved
                MedlinePlus context. It is not a confirmed diagnosis.
                Seek professional medical care when appropriate.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### ⬇️ Export Report")

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "Download JSON Report",
                data=report_json(report),
                file_name=(
                    f"{report['report_id']}.json"
                ),
                mime="application/json",
                use_container_width=True,
            )

        with d2:
            st.download_button(
                "Download Text Report",
                data=report_text(report),
                file_name=(
                    f"{report['report_id']}.txt"
                ),
                mime="text/plain",
                use_container_width=True,
            )

    else:
        st.info(
            "Enter the patient's age, gender and symptoms, "
            "then click **Analyze & Generate Medical Report**."
        )


# ============================================================
# PAGE 2 — FULL MEDICAL REPORT
# ============================================================

elif page == "📋 Medical Report":

    report = st.session_state.current_report

    if not report:
        st.info(
            "No report is available yet. Generate a report from "
            "the Medical Dashboard."
        )
        st.stop()

    analysis = report["nlp_analysis"]

    st.markdown("## 📋 Medical Information Report")

    st.caption(
        f"Report ID: {report['report_id']} • "
        f"{report['created_at']}"
    )

    st.markdown("### 1. Patient Profile")

    a, b = st.columns(2)

    with a:
        st.metric(
            "Age",
            report["patient"]["age"],
        )

    with b:
        st.metric(
            "Gender",
            report["patient"]["gender"],
        )

    st.markdown("### 2. Reported Complaint")

    st.markdown(
        f"""
        <div class="card">
            {escape(report['original_question'])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 3. NLP Symptom Findings")

    present = analysis.get(
        "present_symptoms", []
    )

    if present:
        for item in present:
            symptom = item.get(
                "symptom",
                "Unknown",
            )
            severity = item.get(
                "severity",
                "Not specified",
            )

            st.write(
                f"• **{symptom}** — "
                f"severity: **{severity}**"
            )
    else:
        st.info("No present symptoms detected.")

    absent = analysis.get(
        "absent_symptoms", []
    )

    if absent:
        st.markdown("#### Negated / Absent")

        for item in absent:
            st.write(
                f"• {item.get('symptom', 'Unknown')}"
            )

    duration = analysis.get(
        "duration",
        [],
    )

    st.markdown("#### Duration")

    if duration:
        for value in duration:
            st.write(f"• {value}")
    else:
        st.write("Not detected")

    st.markdown("### 4. AI Medical Information")

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        answer_to_html(
            report["ai_response"]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 5. Important Safety Note")

    st.markdown(
        """
        <div class="warning">
            This report summarizes reported symptoms and information
            retrieved from the configured medical knowledge source.
            It should not be treated as a confirmed diagnosis or as
            a substitute for examination by a qualified healthcare
            professional.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 3 — NLP ANALYSIS
# ============================================================

elif page == "🧠 NLP Analysis":

    report = st.session_state.current_report

    if not report:
        st.info("Generate a report first.")
        st.stop()

    analysis = report["nlp_analysis"]

    st.markdown("## 🧠 NLP Symptom Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Present",
            len(
                analysis.get(
                    "present_symptoms",
                    [],
                )
            ),
        )

    with c2:
        st.metric(
            "Absent",
            len(
                analysis.get(
                    "absent_symptoms",
                    [],
                )
            ),
        )

    with c3:
        st.metric(
            "Duration",
            len(
                analysis.get(
                    "duration",
                    [],
                )
            ),
        )

    st.markdown("### Extracted Symptoms")

    rows = []

    for item in analysis.get(
        "present_symptoms",
        [],
    ):
        rows.append(
            {
                "Symptom": item.get(
                    "symptom",
                    "",
                ),
                "Status": "Present",
                "Severity": item.get(
                    "severity",
                    "Not specified",
                ),
            }
        )

    for item in analysis.get(
        "absent_symptoms",
        [],
    ):
        rows.append(
            {
                "Symptom": item.get(
                    "symptom",
                    "",
                ),
                "Status": "Absent",
                "Severity": "-",
            }
        )

    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No symptoms detected.")

    st.markdown("### Duration")

    duration = analysis.get(
        "duration",
        [],
    )

    if duration:
        for item in duration:
            st.write(f"• {item}")
    else:
        st.write("No duration detected.")

    st.markdown("### spaCy Entities")

    entities = analysis.get(
        "spacy_entities",
        [],
    )

    if entities:
        st.dataframe(
            [
                {
                    "Text": x.get("text", ""),
                    "Label": x.get("label", ""),
                    "Start": x.get("start", ""),
                    "End": x.get("end", ""),
                }
                for x in entities
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No general spaCy entities detected.")

    st.markdown("### Original Query")

    st.code(
        report["original_question"],
        language="text",
    )


# ============================================================
# PAGE 4 — RETRIEVED SOURCES
# ============================================================

elif page == "📚 Retrieved Sources":

    report = st.session_state.current_report

    if not report:
        st.info("Generate a report first.")
        st.stop()

    st.markdown("## 📚 Retrieved Medical Sources")

    st.caption(
        "These are the knowledge-base topics used by the RAG pipeline."
    )

    sources = report.get(
        "sources",
        [],
    )

    if not sources:
        st.info("No sources were retrieved.")
        st.stop()

    for index, source in enumerate(
        sources,
        start=1,
    ):
        title = escape(
            str(source.get("title", "Unknown"))
        )

        url = source.get(
            "url",
            "",
        )

        source_name = escape(
            str(source.get(
                "source",
                "MedlinePlus",
            ))
        )

        if url:
            link = (
                f'<a href="{escape(url)}" '
                f'target="_blank">Open source ↗</a>'
            )
        else:
            link = "No URL available"

        st.markdown(
            f"""
            <div class="source">
                <b>{index}. {title}</b><br>
                <span style="color:#667085">
                    {source_name}
                </span><br><br>
                {link}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 5 — HISTORY
# ============================================================

elif page == "🕘 Report History":

    st.markdown("## 🕘 Consultation History")

    history = st.session_state.report_history

    if not history:
        st.info(
            "No reports have been generated in this Streamlit session."
        )
        st.stop()

    st.write(
        f"Reports in current session: **{len(history)}**"
    )

    for index, report in enumerate(
        history,
        start=1,
    ):

        analysis = report["nlp_analysis"]

        symptoms = [
            item.get("symptom", "")
            for item in analysis.get(
                "present_symptoms",
                [],
            )
        ]

        title = (
            f"{index}. {report['report_id']} — "
            f"{', '.join(symptoms) if symptoms else 'No symptoms detected'}"
        )

        with st.expander(title):

            st.write(
                f"**Created:** {report['created_at']}"
            )

            st.write(
                f"**Age:** {report['patient']['age']}"
            )

            st.write(
                f"**Gender:** {report['patient']['gender']}"
            )

            st.write(
                f"**Query:** {report['original_question']}"
            )

            st.download_button("Download this report",data=report_json(report),file_name=f"{report['report_id']}.json",mime="application/json",
                key=f"download_{report['report_id']}",
            )

