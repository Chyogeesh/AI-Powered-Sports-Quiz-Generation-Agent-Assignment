"""
The RAG "brain" of the agent:
  1. Pulls relevant historic facts from ChromaDB.
  2. Pulls fresh news snippets from DuckDuckGo web search.
  3. Merges both into a single grounding context.
  4. Sends a structured prompt to the LLM and returns parsed quiz questions.
"""

import re
from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.database import query_historic_facts
from src.search import get_live_news_context


def _build_context(sport: str) -> str:
    db_query = f"{sport} history cup championships rules records"
    db_matches = query_historic_facts(sport=sport, query_text=db_query, n_results=3)
    db_context = "\n".join(f"- {m}" for m in db_matches) if db_matches else "No offline historic data found."

    web_context = get_live_news_context(sport)

    return (
        f"=== HISTORICAL FACTS (ChromaDB) ===\n{db_context}\n\n"
        f"=== LIVE INTERNET NEWS (DuckDuckGo) ===\n{web_context}"
    )


def _build_prompt(sport: str, difficulty: str, num_questions: int, context: str):
    system_instruction = (
        "You are an expert sports quiz creator. Write multiple-choice quiz questions "
        "relying strictly on the provided Context. Never invent facts that are not "
        "supported by the Context. If the Context is sparse, write fewer confident "
        "questions rather than guessing.\n\n"
        f"CONTEXT:\n{context}"
    )

    user_prompt = (
        f"Generate exactly {num_questions} unique multiple-choice questions for the sport: {sport}.\n"
        f"Difficulty target: {difficulty}.\n\n"
        "Format EACH question exactly like this, with '---' as a separator between questions "
        "(this exact format is required so a program can parse it):\n\n"
        "Question: [question text]\n"
        "A) [option A]\n"
        "B) [option B]\n"
        "C) [option C]\n"
        "D) [option D]\n"
        "Correct Answer: [single letter, e.g. A]\n"
        "Explanation: [short explanation grounded in the context]\n"
        "---"
    )
    return system_instruction, user_prompt


def parse_quiz_text(raw_text: str):
    """
    Parses the LLM's structured text output into a list of dicts:
    [{question, options: {A,B,C,D}, correct, explanation}, ...]
    Falls back gracefully if a block doesn't match the expected shape.
    """
    blocks = [b.strip() for b in raw_text.split("---") if b.strip()]
    parsed = []

    for block in blocks:
        q_match = re.search(r"Question:\s*(.+)", block)
        a_match = re.search(r"A\)\s*(.+)", block)
        b_match = re.search(r"B\)\s*(.+)", block)
        c_match = re.search(r"C\)\s*(.+)", block)
        d_match = re.search(r"D\)\s*(.+)", block)
        correct_match = re.search(r"Correct Answer:\s*([A-D])", block)
        explanation_match = re.search(r"Explanation:\s*(.+)", block, re.DOTALL)

        if not (q_match and a_match and b_match and c_match and d_match and correct_match):
            continue

        parsed.append({
            "question": q_match.group(1).strip(),
            "options": {
                "A": a_match.group(1).strip(),
                "B": b_match.group(1).strip(),
                "C": c_match.group(1).strip(),
                "D": d_match.group(1).strip(),
            },
            "correct": correct_match.group(1).strip(),
            "explanation": explanation_match.group(1).strip() if explanation_match else "",
        })

    return parsed


def compile_quiz_data(sport: str, difficulty: str, num_questions: int = 4):
    """
    Full pipeline: gather context -> call LLM -> parse structured questions.
    Returns (parsed_questions, raw_text, context_used).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file before generating a quiz."
        )

    context = _build_context(sport)
    system_instruction, user_prompt = _build_prompt(sport, difficulty, num_questions, context)

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    raw_text = response.choices[0].message.content
    parsed_questions = parse_quiz_text(raw_text)

    return parsed_questions, raw_text, context
