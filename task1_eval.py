import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def load_golden_dataset(filepath="data/golden.jsonl"):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def run_eval(version: str):
    prompt_file = f"prompts/classifier_{version}.txt"
    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()

    golden_set = load_golden_dataset()
    correct = 0
    total = len(golden_set)

    print(f"\n Running Evaluation for {version.upper()} ")
    for item in golden_set:
        ticket = item["ticket"]
        expected = item["category"]

        prompt = template.format(ticket=ticket)

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            raw_prediction = response.text.strip().replace(".", "")
            prediction = "Unknown"
            for cat in ["Billing", "Technical", "Account"]:
                if cat.lower() in raw_prediction.lower():
                    prediction = cat
                    break
        except Exception as e:
            prediction = "Error"

        is_match = (prediction == expected)
        if is_match:
            correct += 1
        
        status_sym = "True" if is_match else "False"
        print(f"{status_sym} Expected: {expected:<10} | Predicted: {prediction:<10} | Ticket: {ticket[:45]}...")

    acc_pct = int(round((correct / total) * 100))
    print(f"\nResult: {version}: {correct}/{total} ({acc_pct}%)")
    return correct, total

if __name__ == "__main__":
    run_eval("v1")
    run_eval("v2")