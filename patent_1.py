import json
from pathlib import Path
import ollama  # pip install ollama
import os
MODEL_NAME = "goekdenizguelmez/JOSIEFIED-Qwen3:latest"
FILE_PATH = Path("Webtext/Webtext")
target_pages = []
def classify_text_is_plastic_recycling(text: str) -> str:
    """
    Ask the model to decide if the text contains information of goods and/or services
    related to plastic recycling. Returns 'yes' or 'no'.
    """
    prompt = (
        "Decide if the following text contains information about goods and/or services "
        "related to plastic recycling. Answer with exactly 'yes' or 'no'.\n\n"
        f"Text:\n{text}"
    )

    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    # Ollama returns {'model': ..., 'created_at': ..., 'message': {'role': 'assistant', 'content': '...'}, ...}
    content = resp.get("message", {}).get("content", "").strip().lower()
    # Normalize to just 'yes' or 'no'
    answer = "yes" if content.startswith("yes") else "no"
    return answer

def main():

    files = os.listdir(FILE_PATH)
    for file in files:
        print(f"Found JSON file: {file}")
        file_path = FILE_PATH / file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "description" in data:
            pass
        else:
            print("No 'description' field found in JSON; skipping file.")
            continue

        # Choose which field to analyze; fallback to description if 'text' is missing
        text_to_check = data.get("text") or data.get("description") or ""
        if not text_to_check:
            print("No 'text' or 'description' field found in JSON; nothing to analyze.")
            return

        # Ask the model
        answer = classify_text_is_plastic_recycling(text_to_check)
        print("Model answer:", answer)

        if answer == "yes":
            target_pages.append(file)
            print("The text contains goods and/or services related to plastic recycling.")
        else:
            print("The text does not contain goods and/or services related to plastic recycling.")
        print("-" * 40)
if __name__ == "__main__":
    main()
