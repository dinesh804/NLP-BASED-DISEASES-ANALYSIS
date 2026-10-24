from pathlib import Path
import json

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ============================================================
# SYMPTOM ANALYSIS MODULE
# ============================================================

from symptom import analyze_query


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# Assumes rag_pipeline.py is inside:
# F:\NLP based dieses analysis\
#
# If your file is inside another folder, change BASE_DIR accordingly.

BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# MedlinePlus English JSON dataset
# ------------------------------------------------------------

DATASET_PATH = Path(
    r"F:\NLP based dieses analysis\medlineplus_english_topics (1).json"
)


# ------------------------------------------------------------
# FAISS vector database
# ------------------------------------------------------------

VECTORSTORE_PATH = (
    BASE_DIR / "vectorstore" / "medlineplus_faiss"
)


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ------------------------------------------------------------
# Ollama model
# ------------------------------------------------------------

OLLAMA_MODEL = "llama3.2"


# ------------------------------------------------------------
# Final number of unique medical topics
# ------------------------------------------------------------

TOP_K = 5


# ------------------------------------------------------------
# Number of FAISS candidates retrieved before deduplication
# ------------------------------------------------------------

RETRIEVAL_CANDIDATES = 12


# ============================================================
# 2. CHECK DATASET
# ============================================================

def check_dataset():
    """Validate that the configured MedlinePlus dataset exists."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )
    return DATASET_PATH


# ============================================================
# 3. LOAD MEDLINEPLUS JSON
# ============================================================

def load_medlineplus():

    print("\n" + "=" * 70)
    print("STEP 1 - LOADING MEDLINEPLUS DATASET")
    print("=" * 70)

    loader = JSONLoader(
        file_path=str(DATASET_PATH),
        jq_schema=".[]",
        text_content=False,
    )

    raw_documents = loader.load()

    print(
        f"Loaded {len(raw_documents)} JSON records."
    )

    return raw_documents


# ============================================================
# 4. CONVERT JSON → LANGCHAIN DOCUMENTS
# ============================================================

def prepare_documents(raw_documents):

    print("\n" + "=" * 70)
    print("STEP 2 - PREPARING LANGCHAIN DOCUMENTS")
    print("=" * 70)

    documents = []

    for index, raw_doc in enumerate(
        raw_documents,
        start=1
    ):

        try:

            record = json.loads(
                raw_doc.page_content
            )

            # ------------------------------------------------
            # Basic fields
            # ------------------------------------------------

            title = record.get(
                "title",
                ""
            )

            summary = record.get(
                "summary",
                ""
            )

            url = record.get(
                "url",
                ""
            )

            also_called = record.get(
                "also_called",
                []
            )

            groups = record.get(
                "groups",
                []
            )

            mesh_terms = record.get(
                "mesh_terms",
                []
            )

            related_topics = record.get(
                "related_topics",
                []
            )

            primary_institute = record.get(
                "primary_institute",
                ""
            )

            # ------------------------------------------------
            # Also called
            # ------------------------------------------------

            if isinstance(
                also_called,
                list
            ):

                also_called_text = ", ".join(
                    str(x)
                    for x in also_called
                )

            else:

                also_called_text = str(
                    also_called
                )

            # ------------------------------------------------
            # Groups
            # ------------------------------------------------

            group_names = []

            if isinstance(
                groups,
                list
            ):

                for item in groups:

                    if isinstance(
                        item,
                        dict
                    ):

                        name = item.get(
                            "name",
                            ""
                        )

                        if name:

                            group_names.append(
                                name
                            )

                    else:

                        group_names.append(
                            str(item)
                        )

            groups_text = ", ".join(
                group_names
            )

            # ------------------------------------------------
            # MeSH terms
            # ------------------------------------------------

            mesh_names = []

            if isinstance(
                mesh_terms,
                list
            ):

                for item in mesh_terms:

                    if isinstance(
                        item,
                        dict
                    ):

                        term = item.get(
                            "term",
                            ""
                        )

                        if term:

                            mesh_names.append(
                                term
                            )

                    else:

                        mesh_names.append(
                            str(item)
                        )

            mesh_text = ", ".join(
                mesh_names
            )

            # ------------------------------------------------
            # Related topics
            # ------------------------------------------------

            related_names = []

            if isinstance(
                related_topics,
                list
            ):

                for item in related_topics:

                    if isinstance(
                        item,
                        dict
                    ):

                        related_title = item.get(
                            "title",
                            ""
                        )

                        if related_title:

                            related_names.append(
                                related_title
                            )

                    else:

                        related_names.append(
                            str(item)
                        )

            related_text = ", ".join(
                related_names
            )

            # ------------------------------------------------
            # Final document text
            # ------------------------------------------------

            page_content = f"""
Title:
{title}

Also Called:
{also_called_text}

Summary:
{summary}

Medical Subject Headings:
{mesh_text}

Health Topic Groups:
{groups_text}

Related Topics:
{related_text}

Primary Institute:
{primary_institute}

Source URL:
{url}
""".strip()

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {

                "title": title,

                "url": url,

                "source":
                    "MedlinePlus / U.S. National Library of Medicine",

                "document_id":
                    index,
            }

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata
                )
            )

        except Exception as error:

            print(
                f"Error processing record "
                f"{index}: {error}"
            )

    print(
        f"Created {len(documents)} LangChain documents."
    )

    return documents


# ============================================================
# 5. CHUNK DOCUMENTS
# ============================================================

def create_chunks(documents):

    print("\n" + "=" * 70)
    print("STEP 3 - CHUNKING DOCUMENTS")
    print("=" * 70)

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Original documents : {len(documents)}"
    )

    print(
        f"Generated chunks   : {len(chunks)}"
    )

    if chunks:

        print(
            f"\nFirst chunk size: "
            f"{len(chunks[0].page_content)} characters"
        )

        print(
            "\nFirst chunk:"
        )

        print(
            chunks[0].page_content
        )

    return chunks


# ============================================================
# 6. CREATE EMBEDDINGS
# ============================================================

def create_embeddings():

    print("\n" + "=" * 70)
    print("STEP 4 - LOADING EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    embeddings = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print(
        "Embedding model loaded successfully."
    )

    return embeddings


# ============================================================
# 7. CREATE FAISS VECTOR DATABASE
# ============================================================

def create_vectorstore(
    chunks,
    embeddings
):

    print("\n" + "=" * 70)
    print("STEP 5 - CREATING FAISS VECTOR DATABASE")
    print("=" * 70)

    print(
        "Creating embeddings for chunks..."
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    VECTORSTORE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTORSTORE_PATH)
    )

    print(
        "\nFAISS vector database created."
    )

    print(
        f"Saved at:\n{VECTORSTORE_PATH}"
    )

    return vectorstore


# ============================================================
# 8. LOAD EXISTING FAISS DATABASE
# ============================================================

def load_vectorstore(
    embeddings
):

    print("\n" + "=" * 70)
    print("LOADING EXISTING FAISS DATABASE")
    print("=" * 70)

    vectorstore = FAISS.load_local(

        str(VECTORSTORE_PATH),

        embeddings,

        allow_dangerous_deserialization=True
    )

    print(
        "FAISS database loaded successfully."
    )

    return vectorstore


# ============================================================
# 9. CREATE RETRIEVER
# ============================================================

def create_retriever(
    vectorstore
):

    print("\n" + "=" * 70)
    print("STEP 6 - CREATING RETRIEVER")
    print("=" * 70)

    retriever = vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": RETRIEVAL_CANDIDATES
        }
    )

    print(
        "Retriever created."
    )

    print(
        f"Candidate documents: "
        f"{RETRIEVAL_CANDIDATES}"
    )

    print(
        f"Final unique topics: "
        f"{TOP_K}"
    )

    return retriever


# ============================================================
# 10. LOAD OLLAMA
# ============================================================

def create_llm():

    print("\n" + "=" * 70)
    print("STEP 7 - LOADING OLLAMA LLM")
    print("=" * 70)

    print(
        f"Ollama model: {OLLAMA_MODEL}"
    )

    # The default Ollama server is local.
    # Keeping the base URL explicit makes the Streamlit connection
    # easier to diagnose.
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url="http://127.0.0.1:11434",
    )

    print(
        "Ollama LLM initialized."
    )

    return llm


# ============================================================
# 11. SYMPTOM ANALYSIS DISPLAY
# ============================================================

def display_symptom_analysis(
    analysis
):

    print("\n" + "=" * 70)
    print("NLP SYMPTOM ANALYSIS")
    print("=" * 70)

    if not analysis:

        print(
            "No symptom analysis available."
        )

        return

    # --------------------------------------------------------
    # Present symptoms
    # --------------------------------------------------------

    present_symptoms = analysis.get(
        "present_symptoms",
        []
    )

    print("\nPresent Symptoms:")

    if present_symptoms:

        for item in present_symptoms:

            if isinstance(
                item,
                dict
            ):

                symptom = item.get(
                    "symptom",
                    ""
                )

                status = item.get(
                    "status",
                    "present"
                )

                severity = item.get(
                    "severity",
                    None
                )

                print(
                    f"  - {symptom}"
                    f" | status={status}"
                    f" | severity={severity}"
                )

            else:

                print(
                    f"  - {item}"
                )

    else:

        print(
            "  None detected."
        )

    # --------------------------------------------------------
    # Absent symptoms
    # --------------------------------------------------------

    absent_symptoms = analysis.get(
        "absent_symptoms",
        []
    )

    print("\nAbsent Symptoms:")

    if absent_symptoms:

        for item in absent_symptoms:

            if isinstance(
                item,
                dict
            ):

                symptom = item.get(
                    "symptom",
                    ""
                )

                print(
                    f"  - {symptom}"
                )

            else:

                print(
                    f"  - {item}"
                )

    else:

        print(
            "  None detected."
        )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    durations = analysis.get(
        "duration",
        []
    )

    print("\nDuration:")

    if durations:

        for duration in durations:

            print(
                f"  - {duration}"
            )

    else:

        print(
            "  None detected."
        )


# ============================================================
# 12. COLLECT PATIENT PROFILE
# ============================================================

def collect_patient_profile():
    """Collect age and gender after the user enters a health question."""

    print("\n" + "=" * 70)
    print("PATIENT PROFILE")
    print("=" * 70)

    while True:
        age_input = input(
            "\nEnter age in years: "
        ).strip()

        try:
            age = int(age_input)

            if 0 <= age <= 120:
                break

            print("Please enter an age between 0 and 120.")

        except ValueError:
            print("Please enter a valid numeric age.")

    print("\nSelect gender:")
    print("1. Male")
    print("2. Female")
    print("3. Other")
    print("4. Prefer not to say")

    gender_map = {
        "1": "Male",
        "2": "Female",
        "3": "Other",
        "4": "Prefer not to say"
    }

    while True:
        choice = input(
            "\nEnter choice (1-4): "
        ).strip()

        if choice in gender_map:
            gender = gender_map[choice]
            break

        print("Please select 1, 2, 3, or 4.")

    profile = {
        "age": age,
        "gender": gender
    }

    print("\nPatient profile:")
    print(f"  Age    : {age}")
    print(f"  Gender : {gender}")

    return profile


# ============================================================
# 13. BUILD ENHANCED QUERY
# ============================================================


def build_enhanced_query(
    original_question,
    analysis,
    patient_profile
):
    """
    Build a structured retrieval query from patient profile
    and NLP-extracted symptom information.
    """

    query_parts = []

    # --------------------------------------------------------
    # Patient profile
    # --------------------------------------------------------

    query_parts.append(
        f"Patient age: {patient_profile.get('age', 'unknown')}"
    )

    query_parts.append(
        f"Patient gender: {patient_profile.get('gender', 'unknown')}"
    )

    # --------------------------------------------------------
    # Original query
    # --------------------------------------------------------

    query_parts.append(
        f"Original query: {original_question}"
    )

    # --------------------------------------------------------
    # Present symptoms
    # --------------------------------------------------------

    present_symptoms = analysis.get(
        "present_symptoms",
        []
    )

    present_names = []

    for item in present_symptoms:

        if isinstance(item, dict):

            symptom = item.get(
                "symptom",
                ""
            )

            if symptom:
                present_names.append(symptom)

        elif item:

            present_names.append(str(item))

    if present_names:

        query_parts.append(
            "Present symptoms: "
            + ", ".join(present_names)
        )

    # --------------------------------------------------------
    # Absent symptoms
    # --------------------------------------------------------

    absent_symptoms = analysis.get(
        "absent_symptoms",
        []
    )

    absent_names = []

    for item in absent_symptoms:

        if isinstance(item, dict):

            symptom = item.get(
                "symptom",
                ""
            )

            if symptom:
                absent_names.append(symptom)

        elif item:

            absent_names.append(str(item))

    if absent_names:

        query_parts.append(
            "Absent symptoms: "
            + ", ".join(absent_names)
        )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    durations = analysis.get(
        "duration",
        []
    )

    if durations:

        query_parts.append(
            "Duration: "
            + ", ".join(str(x) for x in durations)
        )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity_items = []

    for item in present_symptoms:

        if isinstance(item, dict):

            symptom = item.get(
                "symptom",
                ""
            )

            severity = item.get(
                "severity",
                None
            )

            if symptom and severity:

                severity_items.append(
                    f"{symptom}: {severity}"
                )

    if severity_items:

        query_parts.append(
            "Severity: "
            + ", ".join(severity_items)
        )

    enhanced_query = "\n".join(query_parts)

    print("\n" + "=" * 70)
    print("ENHANCED RETRIEVAL QUERY")
    print("=" * 70)
    print(enhanced_query)

    return enhanced_query


# ============================================================
# 14. DEDUPLICATE RETRIEVED DOCUMENTS
# ============================================================

def deduplicate_documents(
    documents,
    max_documents=TOP_K
):

    """
    FAISS may return multiple chunks from the
    same MedlinePlus topic.

    Example:

        Nausea and Vomiting
        Nausea and Vomiting
        Nausea and Vomiting
        Gastroenteritis

    This function keeps only one document
    per unique medical topic.
    """

    unique_documents = []

    seen_titles = set()

    for document in documents:

        title = document.metadata.get(
            "title",
            ""
        )

        normalized_title = (
            title.strip().lower()
        )

        # If title is missing, use document ID
        if not normalized_title:

            normalized_title = (
                f"document_"
                f"{document.metadata.get('document_id', '')}"
            )

        if normalized_title in seen_titles:

            continue

        seen_titles.add(
            normalized_title
        )

        unique_documents.append(
            document
        )

        if len(unique_documents) >= max_documents:

            break

    return unique_documents


# ============================================================
# 15. RETRIEVE MEDICAL INFORMATION
# ============================================================

def retrieve_context(
    retriever,
    question
):

    print("\n" + "=" * 70)
    print("STEP 8 - RETRIEVING MEDICAL CONTEXT")
    print("=" * 70)

    # --------------------------------------------------------
    # Retrieve more candidates first
    # --------------------------------------------------------

    retrieved_documents = retriever.invoke(
        question
    )

    print(
        f"Retrieved candidate documents: "
        f"{len(retrieved_documents)}"
    )

    # --------------------------------------------------------
    # Remove duplicate topics
    # --------------------------------------------------------

    unique_documents = deduplicate_documents(
        retrieved_documents,
        max_documents=TOP_K
    )

    print(
        f"Unique medical topics selected: "
        f"{len(unique_documents)}"
    )

    # --------------------------------------------------------
    # Display selected topics
    # --------------------------------------------------------

    print("\nSelected MedlinePlus topics:")

    for index, document in enumerate(
        unique_documents,
        start=1
    ):

        title = document.metadata.get(
            "title",
            "Unknown"
        )

        print(
            f"  {index}. {title}"
        )

    return unique_documents


# ============================================================
# 16. FORMAT CONTEXT
# ============================================================

def format_context(
    retrieved_documents
):

    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        title = document.metadata.get(
            "title",
            "Unknown"
        )

        url = document.metadata.get(
            "url",
            ""
        )

        content = document.page_content

        context_parts.append(

            f"""
SOURCE {index}

Title:
{title}

URL:
{url}

Content:
{content}
""".strip()

        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# 17. FORMAT NLP ANALYSIS FOR LLM
# ============================================================

def format_analysis_for_llm(
    analysis,
    patient_profile
):
    """Convert patient profile and NLP analysis to JSON for Llama."""

    structured_analysis = {
        "patient_profile": {
            "age": patient_profile.get("age"),
            "gender": patient_profile.get("gender")
        },
        "present_symptoms": [],
        "absent_symptoms": [],
        "duration": []
    }

    # --------------------------------------------------------
    # Present symptoms
    # --------------------------------------------------------

    for item in analysis.get(
        "present_symptoms",
        []
    ):

        if isinstance(item, dict):

            structured_analysis[
                "present_symptoms"
            ].append({
                "symptom": item.get(
                    "symptom",
                    ""
                ),
                "status": item.get(
                    "status",
                    "present"
                ),
                "severity": item.get(
                    "severity",
                    None
                )
            })

        else:

            structured_analysis[
                "present_symptoms"
            ].append({
                "symptom": str(item),
                "status": "present",
                "severity": None
            })

    # --------------------------------------------------------
    # Absent symptoms
    # --------------------------------------------------------

    for item in analysis.get(
        "absent_symptoms",
        []
    ):

        if isinstance(item, dict):

            structured_analysis[
                "absent_symptoms"
            ].append({
                "symptom": item.get(
                    "symptom",
                    ""
                ),
                "status": "absent"
            })

        else:

            structured_analysis[
                "absent_symptoms"
            ].append({
                "symptom": str(item),
                "status": "absent"
            })

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    structured_analysis["duration"] = [
        str(x)
        for x in analysis.get(
            "duration",
            []
        )
    ]

    return json.dumps(
        structured_analysis,
        indent=2
    )


# ============================================================
# 18. RAG PROMPT
# ============================================================
def create_prompt():

    prompt_template = """

You are a concise medical information and decision-support assistant.

Use ONLY the retrieved MedlinePlus context. Give a short, clear,
easy-to-understand answer for the user's reported symptoms.

IMPORTANT:
- This is medical information, not a diagnosis.
- Do not say the patient definitely has a disease.
- Do not invent facts, symptoms, medicines, dosages, remedies,
  tests, or treatments.
- Do not turn one symptom into another. For example, stomach pain
  is NOT diarrhea unless the user reported diarrhea.
- Do not describe an absent/negated symptom as present.
- Preserve the reported duration and severity.
- Age and gender are context only; never use them alone to claim
  a disease.
- Possible conditions must come only from the retrieved context.
- Home/self-care advice must come only from the retrieved context.
- If the retrieved context does not support home care, say:
  "No specific home-care advice was found in the retrieved
   MedlinePlus information."
- If evidence is insufficient, say so briefly.
- Prefer 1-3 relevant possible conditions instead of a long list.
- Keep the complete answer SHORT and practical.

RESPONSE FORMAT

1. SYMPTOMS
Briefly state the age, gender, reported symptoms, and duration.

2. POSSIBLE CONDITIONS
Give 1-3 possible conditions.
For each, give ONE short reason based on the retrieved context.
Use "may be related to" or "one possibility is".
Never give a confirmed diagnosis.

3. WHAT TO DO
If supported by MedlinePlus, give 2-4 short self-care points.
If the retrieved context does not support home care, say:
"No specific home-care advice was found in the retrieved
MedlinePlus information."

4. WHEN TO GET MEDICAL HELP
Give only the important doctor/urgent warning signs supported by
the retrieved MedlinePlus context.
Keep this section short.

5. NOTE
End with one short sentence:
"This information does not replace advice from a healthcare professional."

OUTPUT STYLE
- Use simple English.
- Use short sentences.
- Use bullets.
- Avoid unnecessary medical terminology; briefly explain a term
  if it is necessary.
- Do not repeat the same information.
- Do not include a long explanation of the sources.
- Answer the user's actual symptoms, not unrelated retrieved topics.

============================================================
PATIENT PROFILE + NLP ANALYSIS
============================================================

{analysis}

============================================================
RETRIEVED MEDLINEPLUS CONTEXT
============================================================

{context}

============================================================
ORIGINAL USER QUESTION
============================================================

{question}

============================================================
FINAL ANSWER
============================================================

Follow the 5-section format above. Keep it concise and easy to
understand.
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "analysis",
            "context",
            "question"
        ]
    )

    return prompt


# ============================================================
# 19. GENERATE RAG ANSWER
# ============================================================

def generate_answer(
    llm,
    prompt,
    analysis,
    patient_profile,
    context,
    question
):

    print("\n" + "=" * 70)
    print("STEP 9 - GENERATING RAG RESPONSE")
    print("=" * 70)

    analysis_text = format_analysis_for_llm(
        analysis,
        patient_profile
    )

    final_prompt = prompt.format(
        analysis=analysis_text,
        context=context,
        question=question
    )

    response = llm.invoke(
        final_prompt
    )

    return response


# ============================================================
# 20. DISPLAY SOURCES
# ============================================================

def display_sources(
    retrieved_documents
):

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        title = document.metadata.get(
            "title",
            "Unknown"
        )

        url = document.metadata.get(
            "url",
            ""
        )

        print(
            f"\n[{index}] {title}"
        )

        print(
            f"URL: {url}"
        )


# ============================================================
# 21. MAIN RAG PIPELINE
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print("MEDICAL RAG SYSTEM")
    print("MedlinePlus + NLP + LangChain + FAISS + Ollama")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # Load dataset
    # ========================================================

    raw_documents = load_medlineplus()

    # ========================================================
    # STEP 2
    # Prepare documents
    # ========================================================

    documents = prepare_documents(
        raw_documents
    )

    # ========================================================
    # STEP 3
    # Chunk documents
    # ========================================================

    chunks = create_chunks(
        documents
    )

    # ========================================================
    # STEP 4
    # Embeddings
    # ========================================================

    embeddings = create_embeddings()

    # ========================================================
    # STEP 5
    # Create or load FAISS
    # ========================================================

    if VECTORSTORE_PATH.exists():

        print(
            "\nExisting vector database found."
        )

        use_existing = input(
            "\nLoad existing FAISS database? "
            "(y/n): "
        ).strip().lower()

        if use_existing == "y":

            vectorstore = load_vectorstore(
                embeddings
            )

        else:

            vectorstore = create_vectorstore(
                chunks,
                embeddings
            )

    else:

        vectorstore = create_vectorstore(
            chunks,
            embeddings
        )

    # ========================================================
    # STEP 6
    # Retriever
    # ========================================================

    retriever = create_retriever(
        vectorstore
    )

    # ========================================================
    # STEP 7
    # Ollama
    # ========================================================

    llm = create_llm()

    # ========================================================
    # Prompt
    # ========================================================

    prompt = create_prompt()

    # ========================================================
    # Interactive question-answer loop
    # ========================================================

    print("\n" + "=" * 70)
    print("RAG SYSTEM READY")
    print("=" * 70)

    print(
        "\nType your health question."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        print("\n" + "-" * 70)

        # ----------------------------------------------------
        # User symptom query FIRST
        # ----------------------------------------------------

        question = input(
            "\nDescribe your symptoms: "
        ).strip()

        # ----------------------------------------------------
        # Exit BEFORE asking for profile
        # ----------------------------------------------------

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nExiting RAG system."
            )

            break

        # ----------------------------------------------------
        # Empty input
        # ----------------------------------------------------

        if not question:

            print(
                "Please enter a question."
            )

            continue

        # ----------------------------------------------------
        # Patient profile
        # ----------------------------------------------------

        patient_profile = collect_patient_profile()

        # ====================================================
        # STEP A
        # NLP SYMPTOM ANALYSIS
        # ====================================================

        print("\n" + "=" * 70)
        print("STEP 7.5 - NLP SYMPTOM ANALYSIS")
        print("=" * 70)

        try:

            analysis = analyze_query(
                question
            )

        except Exception as error:

            print(
                "\nSymptom analysis failed:"
            )

            print(error)

            analysis = {

                "present_symptoms": [],

                "absent_symptoms": [],

                "duration": []
            }

        # ----------------------------------------------------
        # Safety/grounding fallback for common symptom phrases.
        # The original query remains the source of truth.
        # This only fills missing NLP detections; it does not
        # invent symptoms.
        # ----------------------------------------------------

        if not analysis.get("present_symptoms"):
            common_symptom_phrases = [
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
                ("dizziness", ["dizziness", "dizzy"])
            ]

            question_lower = question.lower()
            fallback_symptoms = []

            for canonical, phrases in common_symptom_phrases:
                if any(phrase in question_lower for phrase in phrases):
                    fallback_symptoms.append({
                        "symptom": canonical,
                        "status": "present",
                        "severity": None
                    })

            if fallback_symptoms:
                analysis["present_symptoms"] = fallback_symptoms

        # ----------------------------------------------------
        # Display NLP analysis
        # ----------------------------------------------------

        display_symptom_analysis(
            analysis
        )

        # ====================================================
        # STEP B
        # BUILD ENHANCED QUERY
        # ====================================================

        enhanced_query = build_enhanced_query(
            question,
            analysis,
            patient_profile
        )

        # ====================================================
        # STEP C
        # RETRIEVE MEDICAL INFORMATION
        # ====================================================

        retrieved_documents = retrieve_context(
            retriever,
            enhanced_query
        )

        # ====================================================
        # STEP D
        # FORMAT CONTEXT
        # ====================================================

        context = format_context(
            retrieved_documents
        )

        # ====================================================
        # STEP E
        # GENERATE ANSWER
        # ====================================================

        answer = generate_answer(

            llm,

            prompt,

            analysis,

            patient_profile,

            context,

            question
        )

        # ====================================================
        # STEP F
        # DISPLAY ANSWER
        # ====================================================

        print("\n" + "=" * 70)
        print("RAG ANSWER")
        print("=" * 70)

        print(
            answer
        )

        # ====================================================
        # STEP G
        # DISPLAY SOURCES
        # ====================================================

        display_sources(
            retrieved_documents
        )


# ============================================================
# 22. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()