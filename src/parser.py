from pathlib import Path
import re


# --------------------------------------------------
# Parse one transcript
# --------------------------------------------------

def parse_transcript(file_path: str) -> list[dict]:

    records = []

    current_timestamp = None
    current_speaker = None
    current_text = []

    timestamp_pattern = re.compile(r"^\d{2}:\d{2}$")
    speaker_pattern = re.compile(r"^([^:]+):\s*(.*)$")

    source = Path(file_path).name

    # Determine country from filename
    if "France" in source:
        country = "France"
    elif "Germany" in source:
        country = "Germany"
    elif "UK" in source:
        country = "UK"
    else:
        country = "Unknown"

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Ignore transcript metadata before interview starts
        if (
            current_timestamp is None
            and line.startswith(("Role:", "Market:"))
        ):
            continue

        # ------------------------------------------
        # New timestamp
        # ------------------------------------------

        if timestamp_pattern.match(line):

            # Save previous record
            if (
                current_timestamp is not None
                and current_speaker is not None
            ):
                records.append({
                    "timestamp": current_timestamp,
                    "speaker": current_speaker,
                    "text": " ".join(current_text),
                    "source": source,
                    "country": country
                })

            current_timestamp = line
            current_speaker = None
            current_text = []

            continue

        # ------------------------------------------
        # Speaker + first part of statement
        # ------------------------------------------

        speaker_match = speaker_pattern.match(line)

        if speaker_match:

            current_speaker = speaker_match.group(1).strip()

            text = speaker_match.group(2).strip()

            if text:
                current_text.append(text)

            continue

        # ------------------------------------------
        # Continuation of previous statement
        # ------------------------------------------

        if current_speaker is not None:
            current_text.append(line)

    # ----------------------------------------------
    # Save final record
    # ----------------------------------------------

    if (
        current_timestamp is not None
        and current_speaker is not None
    ):
        records.append({
            "timestamp": current_timestamp,
            "speaker": current_speaker,
            "text": " ".join(current_text),
            "source": source,
            "country": country
        })

    return records


# --------------------------------------------------
# Load expert records from all transcripts
# --------------------------------------------------

def load_expert_records() -> list[dict]:

    transcript_folder = Path(
        "data/transcripts"
    )

    all_expert_records = []

    for file_path in transcript_folder.glob("*.txt"):

        records = parse_transcript(
            str(file_path)
        )

        expert_records = [
            record
            for record in records
            if record["speaker"] != "Interviewer"
        ]

        all_expert_records.extend(
            expert_records
        )

    return all_expert_records


# --------------------------------------------------
# Test parser
# --------------------------------------------------

if __name__ == "__main__":

    records = load_expert_records()

    print(
        "Total expert statements:",
        len(records)
    )

    for record in records:
        print(record)