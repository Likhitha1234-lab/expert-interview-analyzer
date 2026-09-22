import json
import streamlit as st

from src.rag import answer_question

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Expert Interview Analyzer",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load interview-guide results
# --------------------------------------------------

with open(
    "data/processed/interview_guide_results.json",
    "r",
    encoding="utf-8"
) as file:
    interview_results = json.load(file)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Expert Interview Analyzer")

st.write(
    "Analyze expert interviews, compare markets, "
    "and ask questions using transcript evidence."
)


# ==================================================
# 1. INTERVIEW GUIDE
# ==================================================

st.header("1. Interview Guide")

question_options = {
    result["question"]: result
    for result in interview_results
}

selected_question = st.selectbox(
    "Select a question",
    list(question_options.keys())
)

selected_result = question_options[selected_question]


st.subheader("Answer")

st.write(selected_result["answer"])


st.subheader("Supporting Evidence")

for evidence in selected_result["evidence"]:

    with st.expander(
        f'{evidence["country"]} | '
        f'{evidence["expert"]} | '
        f'{evidence["timestamp"]}'
    ):

        st.write(
            f'**Topic:** {evidence["topic"]}'
        )

        st.write(
            f'**Source:** {evidence["source"]}'
        )

        st.write(
            f'**Exact quote:** "{evidence["quote"]}"'
        )


# ==================================================
# 2. ASK YOUR OWN QUESTION
# ==================================================

st.header("2. Ask Your Own Question")

user_question = st.text_input(
    "Enter your question"
)


if st.button("Ask"):

    if user_question.strip():

        with st.spinner("Analyzing transcripts..."):

            answer, results = answer_question(
                user_question
            )

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Supporting Evidence")

        for i in range(
            len(results["documents"][0])
        ):

            metadata = results["metadatas"][0][i]
            quote = results["documents"][0][i]

            with st.expander(
                f'{metadata["country"]} | '
                f'{metadata["speaker"]} | '
                f'{metadata["timestamp"]}'
            ):

                st.write(
                    f'**Topic:** {metadata["topic"]}'
                )

                st.write(
                    f'**Exact quote:** "{quote}"'
                )

    else:

        st.warning(
            "Please enter a question."
        )