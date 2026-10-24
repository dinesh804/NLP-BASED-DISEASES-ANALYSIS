import re
import json
import spacy


# ============================================================
# CONFIGURATION
# ============================================================

SPACY_MODEL = "en_core_web_sm"


# ============================================================
# LOAD SPACY MODEL
# ============================================================

print("=" * 70)
print("LOADING NLP MODEL")
print("=" * 70)

try:
    nlp = spacy.load(SPACY_MODEL)
    print(f"✓ spaCy model loaded: {SPACY_MODEL}")

except OSError:

    print("\n✗ spaCy model is not installed.")
    print("\nRun:")
    print("python -m spacy download en_core_web_sm")

    raise


# ============================================================
# MEDICAL SYMPTOM VOCABULARY
# ============================================================

SYMPTOM_VOCABULARY = {

    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    "fever": "fever",
    "high temperature": "fever",
    "temperature": "fever",

    "headache": "headache",
    "head pain": "headache",

    "fatigue": "fatigue",
    "tiredness": "fatigue",
    "extreme tiredness": "fatigue",

    "weakness": "weakness",
    "dizziness": "dizziness",
    "fainting": "fainting",

    # --------------------------------------------------------
    # RESPIRATORY
    # --------------------------------------------------------

    "cough": "cough",
    "dry cough": "cough",

    "runny nose": "runny nose",

    "blocked nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "nasal congestion": "nasal congestion",

    "sore throat": "sore throat",

    "shortness of breath": "shortness of breath",
    "difficulty breathing": "difficulty breathing",
    "breathing difficulty": "difficulty breathing",

    "chest pain": "chest pain",

    # --------------------------------------------------------
    # DIGESTIVE
    # --------------------------------------------------------

    "abdominal pain": "abdominal pain",
    "stomach pain": "abdominal pain",
    "belly pain": "abdominal pain",

    "nausea": "nausea",
    "feeling nauseous": "nausea",

    "vomiting": "vomiting",
    "vomit": "vomiting",

    # Common spelling mistake
    "vomitting": "vomiting",

    # Common expressions
    "throwing up": "vomiting",
    "threw up": "vomiting",

    "diarrhea": "diarrhea",
    "diarrhoea": "diarrhea",
    "loose stools": "diarrhea",
    "loose motion": "diarrhea",

    "constipation": "constipation",

    "loss of appetite": "loss of appetite",

    # --------------------------------------------------------
    # PAIN
    # --------------------------------------------------------

    "back pain": "back pain",
    "lower back pain": "back pain",

    "joint pain": "joint pain",
    "muscle pain": "muscle pain",
    "body pain": "body pain",

    # --------------------------------------------------------
    # NEUROLOGICAL
    # --------------------------------------------------------

    "migraine": "migraine",
    "seizure": "seizure",
    "confusion": "confusion",
    "memory loss": "memory loss",

    # --------------------------------------------------------
    # SKIN
    # --------------------------------------------------------

    "rash": "rash",
    "skin rash": "rash",

    "itching": "itching",

    "swelling": "swelling",

    # --------------------------------------------------------
    # EYE
    # --------------------------------------------------------

    "eye itching": "eye itching",
    "itchy eyes": "eye itching",
    "itching eyes": "eye itching",

    "eye pain": "eye pain",
    "pain in the eye": "eye pain",
    "pain in my eye": "eye pain",

    "eye redness": "eye redness",
    "red eyes": "eye redness",
    "red eye": "eye redness",

    "eye swelling": "eye swelling",
    "swollen eye": "eye swelling",

    "watery eyes": "watery eyes",
    "watering eyes": "watery eyes",

    "blurred vision": "blurred vision",
    "blurry vision": "blurred vision",

    "eye discharge": "eye discharge",

    "eye irritation": "eye irritation",
    "eye discomfort": "eye discomfort",

    "light sensitivity": "light sensitivity",
    "sensitivity to light": "light sensitivity",

    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    "weight loss": "weight loss",
    "weight gain": "weight gain",

    "night sweats": "night sweats",
}


# ============================================================
# NEGATION WORDS
# ============================================================

NEGATION_WORDS = {

    "no",
    "not",
    "never",
    "without",

    "don't",
    "dont",

    "doesn't",
    "doesnt",

    "didn't",
    "didnt",

    "haven't",
    "havent",

    "hasn't",
    "hasnt",

    "cannot",
    "can't",
    "cant",
}


# ============================================================
# SEVERITY WORDS
# ============================================================

SEVERITY_WORDS = {

    "mild": "mild",
    "slight": "mild",
    "minor": "mild",

    "moderate": "moderate",

    "severe": "severe",
    "extreme": "severe",
    "intense": "severe",
    "terrible": "severe",
    "very painful": "severe",
}


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_text(text: str) -> str:

    # Convert to lowercase
    text = text.lower()

    # Normalize apostrophes
    text = text.replace("’", "'")

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(text: str):

    doc = nlp(text)

    tokens = []

    for token in doc:

        if not token.is_space:

            tokens.append({

                "text": token.text,

                "lemma": token.lemma_,

                "pos": token.pos_
            })

    return tokens


# ============================================================
# NEGATION DETECTION
# ============================================================

def detect_negation(
    text: str,
    symptom_start: int
) -> str:

    preceding_text = text[
        max(
            0,
            symptom_start - 60
        ):
        symptom_start
    ].lower()

    # --------------------------------------------------------
    # Clause boundaries
    # --------------------------------------------------------

    boundaries = [

        " but ",

        " however ",

        " although ",

        " though ",

        " yet ",

        ";",

        "."
    ]

    for boundary in boundaries:

        if boundary in preceding_text:

            preceding_text = (
                preceding_text
                .rsplit(
                    boundary,
                    1
                )[-1]
            )

    # --------------------------------------------------------
    # Check recent words
    # --------------------------------------------------------

    words = preceding_text.split()

    recent_words = words[-5:]

    for word in recent_words:

        clean_word = re.sub(
            r"[^\w']",
            "",
            word
        )

        if clean_word in NEGATION_WORDS:

            return "absent"

    return "present"


# ============================================================
# SEVERITY DETECTION
# ============================================================

def detect_severity(
    text: str,
    symptom_start: int
):

    preceding_text = text[
        max(
            0,
            symptom_start - 50
        ):
        symptom_start
    ].lower()

    severity_items = sorted(
        SEVERITY_WORDS.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for phrase, severity in severity_items:

        if phrase in preceding_text:

            return severity

    return None


# ============================================================
# DURATION EXTRACTION
# ============================================================

def extract_duration(
    text: str
):

    cleaned_text = preprocess_text(
        text
    )

    durations = []

    # --------------------------------------------------------
    # Primary duration patterns
    # --------------------------------------------------------

    primary_patterns = [

        # for 2 days
        r"\bfor\s+\d+\s+"
        r"(?:day|days|week|weeks|"
        r"month|months|year|years)\b",

        # for a few days
        r"\bfor\s+a\s+few\s+"
        r"(?:days|weeks|months|years)\b",

        # for several weeks
        r"\bfor\s+several\s+"
        r"(?:days|weeks|months|years)\b",

        # since yesterday
        r"\bsince\s+"
        r"(?:yesterday|today|"
        r"last night|last week)\b",

        # since Monday
        r"\bsince\s+\w+\b",
    ]

    for pattern in primary_patterns:

        matches = re.findall(
            pattern,
            cleaned_text
        )

        for match in matches:

            if match not in durations:

                durations.append(
                    match
                )

    # --------------------------------------------------------
    # Standalone duration
    #
    # Only use this when no primary duration
    # expression was found.
    # --------------------------------------------------------

    if not durations:

        standalone_pattern = (

            r"\b\d+\s+"

            r"(?:day|days|week|weeks|"
            r"month|months|year|years)\b"
        )

        matches = re.findall(
            standalone_pattern,
            cleaned_text
        )

        for match in matches:

            if match not in durations:

                durations.append(
                    match
                )

    return durations


# ============================================================
# EYE CONTEXT EXTRACTION
# ============================================================

def extract_eye_symptoms(
    text: str
):

    symptoms = []

    cleaned_text = preprocess_text(
        text
    )

    eye_mapping = {

        "itching":
            "eye itching",

        "itchy":
            "eye itching",

        "pain":
            "eye pain",

        "redness":
            "eye redness",

        "red":
            "eye redness",

        "swelling":
            "eye swelling",

        "discharge":
            "eye discharge",

        "irritation":
            "eye irritation",

        "watering":
            "watery eyes",

        "watery":
            "watery eyes",
    }

    # --------------------------------------------------------
    # Pattern:
    #
    # eye itching and pain
    # eye redness and swelling
    # --------------------------------------------------------

    pattern = (

        r"\b(?:eye|eyes)\s+"

        r"(itching|itchy|pain|redness|red|"
        r"swelling|discharge|irritation|"
        r"watering|watery)\s+"

        r"(?:and|,)\s+"

        r"(itching|itchy|pain|redness|red|"
        r"swelling|discharge|irritation|"
        r"watering|watery)\b"
    )

    match = re.search(
        pattern,
        cleaned_text
    )

    if not match:

        return symptoms

    first = match.group(1)

    second = match.group(2)

    # First symptom
    if first in eye_mapping:

        symptoms.append({

            "name":
                eye_mapping[first],

            "status":
                "present",

            "severity":
                detect_severity(
                    cleaned_text,
                    match.start()
                ),

            "matched_text":
                f"eye {first}",

            "start":
                match.start(),

            "end":
                match.end()
        })

    # Second symptom
    if second in eye_mapping:

        symptoms.append({

            "name":
                eye_mapping[second],

            "status":
                "present",

            "severity":
                detect_severity(
                    cleaned_text,
                    match.start()
                ),

            "matched_text":
                f"eye {second}",

            "start":
                match.start(),

            "end":
                match.end()
        })

    return symptoms


# ============================================================
# SYMPTOM EXTRACTION
# ============================================================

def extract_symptoms(
    text: str
):

    cleaned_text = preprocess_text(
        text
    )

    symptoms = []

    # --------------------------------------------------------
    # Eye-specific extraction
    # --------------------------------------------------------

    symptoms.extend(
        extract_eye_symptoms(
            cleaned_text
        )
    )

    # --------------------------------------------------------
    # General symptom vocabulary
    # --------------------------------------------------------

    sorted_symptoms = sorted(
        SYMPTOM_VOCABULARY.keys(),
        key=len,
        reverse=True
    )

    for symptom_phrase in sorted_symptoms:

        pattern = (

            r"(?<!\w)"

            + re.escape(
                symptom_phrase
            )

            + r"(?!\w)"
        )

        matches = re.finditer(
            pattern,
            cleaned_text
        )

        for match in matches:

            start = match.start()

            end = match.end()

            canonical_name = (
                SYMPTOM_VOCABULARY[
                    symptom_phrase
                ]
            )

            # ------------------------------------------------
            # Avoid duplicates
            # ------------------------------------------------

            already_detected = False

            for existing in symptoms:

                if (
                    existing["name"]
                    == canonical_name
                ):

                    already_detected = True

                    break

            if already_detected:

                continue

            # ------------------------------------------------
            # IMPORTANT:
            # Detect NEGATION here.
            # Do NOT call extract_duration().
            # ------------------------------------------------

            status = detect_negation(
                cleaned_text,
                start
            )

            # ------------------------------------------------
            # Severity
            # ------------------------------------------------

            severity = detect_severity(
                cleaned_text,
                start
            )

            symptoms.append({

                "name":
                    canonical_name,

                "status":
                    status,

                "severity":
                    severity,

                "matched_text":
                    symptom_phrase,

                "start":
                    start,

                "end":
                    end
            })

    return remove_duplicate_symptoms(
        symptoms
    )


# ============================================================
# REMOVE DUPLICATE SYMPTOMS
# ============================================================

def remove_duplicate_symptoms(
    symptoms
):

    unique = {}

    for symptom in symptoms:

        key = (

            symptom["name"],

            symptom["status"],

            symptom["start"]
        )

        if key not in unique:

            unique[key] = symptom

    return list(
        unique.values()
    )


# ============================================================
# SPACY ENTITY EXTRACTION
# ============================================================

def extract_spacy_entities(
    text: str
):

    doc = nlp(text)

    entities = []

    for entity in doc.ents:

        entities.append({

            "text":
                entity.text,

            "label":
                entity.label_,

            "start":
                entity.start_char,

            "end":
                entity.end_char
        })

    return entities


# ============================================================
# COMPLETE QUERY ANALYSIS
# ============================================================

def analyze_query(
    query: str
):

    cleaned_query = preprocess_text(
        query
    )

    symptoms = extract_symptoms(
        cleaned_query
    )

    duration = extract_duration(
        cleaned_query
    )

    spacy_entities = (
        extract_spacy_entities(
            query
        )
    )

    result = {

        "original_query":
            query,

        "cleaned_query":
            cleaned_query,

        "symptoms":
            symptoms,

        "duration":
            duration,

        "spacy_entities":
            spacy_entities
    }

    return result


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    result
):

    print("\n")

    print("=" * 70)

    print(
        "EXTRACTED MEDICAL INFORMATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Original query
    # --------------------------------------------------------

    print("\nOriginal Query:")

    print(
        result[
            "original_query"
        ]
    )

    # --------------------------------------------------------
    # Symptoms
    # --------------------------------------------------------

    print("\nSymptoms:")

    if result["symptoms"]:

        for symptom in result[
            "symptoms"
        ]:

            print(
                f"  • {symptom['name']}"
            )

            print(
                f"      Status   : "
                f"{symptom['status']}"
            )

            print(
                f"      Severity : "
                f"{symptom['severity']}"
            )

            print(
                f"      Matched  : "
                f"{symptom['matched_text']}"
            )

    else:

        print(
            "  No symptoms detected."
        )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    print("\nDuration:")

    if result["duration"]:

        for duration in result[
            "duration"
        ]:

            print(
                f"  • {duration}"
            )

    else:

        print(
            "  No duration detected."
        )

    # --------------------------------------------------------
    # spaCy entities
    # --------------------------------------------------------

    print("\nspaCy Entities:")

    if result[
        "spacy_entities"
    ]:

        for entity in result[
            "spacy_entities"
        ]:

            print(
                f"  • {entity['text']}"
                f" → {entity['label']}"
            )

    else:

        print(
            "  No general entities detected."
        )


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [

    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    "I have fever and severe headache for 3 days.",

    "I have cough and shortness of breath for one week.",

    "I have stomach pain and vomiting but I don't have fever.",

    "I have mild back pain since yesterday.",

    "I have severe chest pain and dizziness.",

    # --------------------------------------------------------
    # Negation
    # --------------------------------------------------------

    "I don't have cough but I have high temperature.",

    "I have no fever, no cough, but severe headache.",

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    "I have diarrhea and vomiting for 2 days.",

    "I have been experiencing moderate abdominal pain for several weeks.",

    # --------------------------------------------------------
    # Spelling variation
    # --------------------------------------------------------

    "I have a vomitting and stomach pain for 2 days.",

    "I have been vomitting for 3 days.",

    # --------------------------------------------------------
    # Eye symptoms
    # --------------------------------------------------------

    "I have eye itching and pain.",

    "I have eye pain and blurred vision.",

    "I have severe eye pain with watery eyes for 2 days.",

    # --------------------------------------------------------
    # Skin
    # --------------------------------------------------------

    "I have a rash and itching for 2 weeks.",
]


# ============================================================
# RUN PREDEFINED TESTS
# ============================================================

def run_tests():

    print("\n")

    print("=" * 70)

    print(
        "SYMPTOM EXTRACTION TESTING"
    )

    print("=" * 70)

    for index, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        print("\n")

        print("#" * 70)

        print(
            f"TEST {index}"
        )

        print(
            "#" * 70
        )

        result = analyze_query(
            query
        )

        display_result(
            result
        )

        print("\nJSON:")

        print(

            json.dumps(

                result,

                indent=4,

                ensure_ascii=False
            )
        )


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode():

    print("\n")

    print("=" * 70)

    print(
        "INTERACTIVE SYMPTOM EXTRACTION"
    )

    print("=" * 70)

    print(
        "\nEnter a health-related query."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        query = input(
            "\nUser: "
        ).strip()

        if query.lower() in {

            "exit",

            "quit",

            "q"
        }:

            print(
                "\nExiting."
            )

            break

        if not query:

            print(
                "Please enter a query."
            )

            continue

        result = analyze_query(
            query
        )

        display_result(
            result
        )

        print("\nJSON:")

        print(

            json.dumps(

                result,

                indent=4,

                ensure_ascii=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "NLP MEDICAL QUERY ANALYZER"
    )

    print("=" * 70)

    print(
        "\nChoose an option:"
    )

    print(
        "1. Run predefined test queries"
    )

    print(
        "2. Enter your own query"
    )

    choice = input(
        "\nEnter choice (1/2): "
    ).strip()

    if choice == "1":

        run_tests()

    elif choice == "2":

        interactive_mode()

    else:

        print(
            "\nInvalid choice."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()