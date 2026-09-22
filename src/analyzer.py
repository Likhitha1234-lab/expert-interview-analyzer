import os
import json
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.parser import load_expert_records


# ==================================================
# 1. Configuration
# ==================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is not set in the .env file"
    )

client = Groq(api_key=api_key)

TOPICS = [
    "Adoption",
    "Barriers",
    "Budget / ROI",
    "Training / Clinical outcomes",
    "Future adoption",
    "Purchasing timeline",
]


# ==================================================
# 2. Interview Guide Questions
# ==================================================

INTERVIEW_QUESTIONS = [
    {
        "id": 1,
        "question": (
            "How would you describe current adoption "
            "of robotic surgery in your market?"
        ),
        "topic": "Adoption",
    },
    {
        "id": 2,
        "question": "What are the main barriers to adoption?",
        "topic": "Barriers",
    },
    {
        "id": 3,
        "question": (
            "How important are hospital budgets and ROI "
            "in purchasing decisions?"
        ),
        "topic": "Budget / ROI",
    },
    {
        "id": 4,
        "question": (
            "How important are surgeon training and "
            "clinical outcomes?"
        ),
        "topic": "Training / Clinical outcomes",
    },
    {
        "id": 5,
        "question": (
            "What adoption trend do you expect "
            "over the next 3–5 years?"
        ),
        "topic": "Future adoption",
    },
    {
        "id": 6,
        "question": (
            "What is the typical hospital decision-making "
            "timeline for purchasing a new robotic system?"
        ),
        "topic": "Purchasing timeline",
    },
]


# ==================================================
# 3. Topic Classification
# ==================================================

def classify_topic(statement: str) -> str:

    prompt = f"""
Classify the following expert statement into exactly
one of these topics:

1. Adoption
2. Barriers
3. Budget / ROI
4. Training / Clinical outcomes
5. Future adoption
6. Purchasing timeline

Expert statement:
{statement}

Return a JSON object with exactly this structure:

{{
    "topic": "one of the six topics above"
}}

Do not include any additional text.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    result = response.choices[0].message.content.strip()

    result_json = json.loads(result)

    topic = result_json["topic"]

    # Handle numeric responses from the model
    if str(topic).isdigit():

        topic_number = int(topic)

        if 1 <= topic_number <= len(TOPICS):
            topic = TOPICS[topic_number - 1]

    if topic not in TOPICS:
        raise ValueError(
            f"Unexpected topic returned by model: {topic}"
        )

    return topic


# ==================================================
# 4. Enrich Transcript Records
# ==================================================

def enrich_records(records: list[dict]) -> list[dict]:

    enriched_records = []

    for index, record in enumerate(
        records,
        start=1
    ):

        print(
            f"Classifying statement "
            f"{index}/{len(records)}..."
        )

        topic = classify_topic(
            record["text"]
        )

        enriched_record = record.copy()

        enriched_record["topic"] = topic

        enriched_records.append(
            enriched_record
        )

    return enriched_records


def save_records(
    records: list[dict],
    output_path: str
):

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nSaved {len(records)} records to: {path}"
    )


# ==================================================
# 5. Interview Guide Analysis
# ==================================================

def load_enriched_records():

    with open(
        "data/processed/expert_records.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def get_topic_records(
    records: list[dict],
    topic: str
):

    return [
        record
        for record in records
        if record["topic"] == topic
    ]


def build_context(records):

    context_parts = []

    for record in records:

        context_parts.append(
            f"""
Country: {record["country"]}
Expert: {record["speaker"]}
Timestamp: {record["timestamp"]}
Topic: {record["topic"]}
Exact quote: {record["text"]}
"""
        )

    return "\n---\n".join(context_parts)


# ==================================================
# 6. Cross-Market Analysis
# ==================================================

def analyze_topic(topic: str):

    records = load_enriched_records()

    topic_records = get_topic_records(
        records,
        topic
    )

    context = build_context(
        topic_records
    )

    prompt = f"""
You are analyzing expert interview transcripts.

Topic:
{topic}

Compare the expert statements across France,
Germany, and the UK.

Identify:

1. Common themes
2. Important differences or disagreements between markets

Use ONLY the evidence provided below.

Do not invent facts.
Do not create quotes.
Do not make claims that are not supported by the evidence.

Keep the analysis concise.

Clearly separate:

- Common themes
- Differences / disagreements

Evidence:

{context}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content


# ==================================================
# 7. Run Interview Guide
# ==================================================

def run_interview_guide():

    records = load_enriched_records()

    all_results = []

    for question_info in INTERVIEW_QUESTIONS:

        question_id = question_info["id"]
        question = question_info["question"]
        topic = question_info["topic"]

        topic_records = get_topic_records(
            records,
            topic
        )

        context = build_context(
            topic_records
        )

        prompt = f"""
You are analyzing expert interview transcripts.

Answer the interview-guide question using ONLY
the evidence provided below.

Do not invent facts.
Do not create quotes.
Do not change the wording of quotes.

Give a concise synthesis of the evidence.

Do NOT provide quotes, timestamps, expert names,
or country names in the answer.

Focus only on the main findings supported
by the evidence.

Interview question:
{question}

Evidence:
{context}
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        answer = response.choices[0].message.content

        evidence = []

        for record in topic_records:

            evidence.append({
                "country": record["country"],
                "expert": record["speaker"],
                "timestamp": record["timestamp"],
                "topic": record["topic"],
                "source": record["source"],
                "quote": record["text"]
            })

        all_results.append({
            "question_id": question_id,
            "question": question,
            "topic": topic,
            "answer": answer,
            "evidence": evidence
        })

    return all_results


# ==================================================
# 8. Save Interview Guide Results
# ==================================================

def save_interview_results(results):

    output_path = Path(
        "data/processed/interview_guide_results.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nSaved interview results to: "
        f"{output_path}"
    )


# ==================================================
# 9. Main
# ==================================================

if __name__ == "__main__":

    # Load and classify transcript records
    records = load_expert_records()

    enriched_records = enrich_records(
        records
    )

    save_records(
        enriched_records,
        "data/processed/expert_records.json"
    )

    # Generate interview-guide answers
    results = run_interview_guide()

    save_interview_results(
        results
    )

    # Example cross-market analysis
    topic = "Budget / ROI"

    analysis = analyze_topic(topic)

    print("\n" + "=" * 70)
    print(
        f"CROSS-MARKET ANALYSIS: {topic}"
    )
    print("=" * 70)

    print(analysis)