import os

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
from src.analyzer import INTERVIEW_QUESTIONS


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the .env file")


# --------------------------------------------------
# 2. Create Groq client
# --------------------------------------------------

groq_client = Groq(api_key=api_key)


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Connect to ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="data/processed/chroma_db"
)

collection = chroma_client.get_collection(
    name="expert_transcripts"
)


# --------------------------------------------------
# 5. Detect topic from user's question
# --------------------------------------------------

def detect_topic(question: str) -> str:

    prompt = f"""
Classify the user's question into exactly one of these topics:

1. Adoption
2. Barriers
3. Budget / ROI
4. Training / Clinical outcomes
5. Future adoption
6. Purchasing timeline

User question:
{question}

Return only the topic name.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    topic = response.choices[0].message.content.strip()

    return topic


# --------------------------------------------------
# 6. RAG function
# --------------------------------------------------

def answer_question(
    question: str,
    expected_topic: str = None
):

    # ----------------------------------------------
    # Detect topic for free-form questions
    # ----------------------------------------------

    if expected_topic is None:
        expected_topic = detect_topic(question)

        print(f"\nDetected topic: {expected_topic}")

    # ----------------------------------------------
    # Convert question into an embedding
    # ----------------------------------------------

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    # ----------------------------------------------
    # Retrieve relevant evidence
    # ----------------------------------------------

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=8,
        where={
            "topic": expected_topic
        }
    )

    # ----------------------------------------------
    # Build context for the LLM
    # ----------------------------------------------

    context_parts = []

    for i in range(len(results["documents"][0])):

        metadata = results["metadatas"][0][i]
        quote = results["documents"][0][i]

        context_parts.append(
            f"""
Country: {metadata["country"]}
Expert: {metadata["speaker"]}
Timestamp: {metadata["timestamp"]}
Topic: {metadata["topic"]}
Exact quote: {quote}
"""
        )

    context = "\n---\n".join(context_parts)

    # ----------------------------------------------
    # Prompt the LLM
    # ----------------------------------------------

    prompt = f"""
You are analyzing expert interview transcripts.

Answer the user's question using ONLY the evidence provided below.

Do not invent facts.
Do not create quotes.
Do not change the wording of quotes.

Give a concise synthesis of the evidence.

Do NOT provide quotes, timestamps, expert names,
or country names in your answer.

Those will be added directly from the original
transcript records by the application.

Focus only on explaining the main findings
supported by the evidence.

User question:
{question}

Detected topic:
{expected_topic}

Evidence:
{context}
"""

    # ----------------------------------------------
    # Generate synthesis
    # ----------------------------------------------

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    answer = response.choices[0].message.content

    return answer, results


# --------------------------------------------------
# 7. Test the function
# --------------------------------------------------

if __name__ == "__main__":

    question = "What are the biggest barriers to adopting robotic surgery?"

    answer, results = answer_question(question)

    print("\n" + "=" * 80)
    print("USER QUESTION")
    print("=" * 80)

    print(question)

    print("\nANSWER")
    print("-" * 80)

    print(answer)

    print("\nSUPPORTING EVIDENCE")
    print("-" * 80)

    for i in range(len(results["documents"][0])):

        metadata = results["metadatas"][0][i]
        quote = results["documents"][0][i]

        print(f"\nEvidence {i + 1}")
        print(f"Country: {metadata['country']}")
        print(f"Expert: {metadata['speaker']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Topic: {metadata['topic']}")
        print(f"Exact quote: {quote}")
        print("-" * 60)