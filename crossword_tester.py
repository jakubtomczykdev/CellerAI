from dotenv import load_dotenv
load_dotenv('.env.local', override=True)
load_dotenv('.env')

import json
import time
import pandas as pd
import re
import os
import litellm
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

litellm.suppress_debug_info = True

MODELS_TO_TEST = [
   "openrouter/openai/gpt-5.4-nano",
    "openrouter/google/gemma-4-31b-it",
    "openrouter/openai/gpt-oss-20b",
    "openrouter/openai/gpt-oss-120b",
#    "openrouter/x-ai/grok-4.20",
    "openrouter/x-ai/grok-4.1-fast",
    "openrouter/google/gemini-3.1-flash-lite-preview",
#     "openrouter/google/gemma-4-26b-a4b-it",
    "openrouter/qwen/qwen3.5-flash-02-23",
    "openrouter/mistralai/ministral-14b-2512",
    "openrouter/mistralai/mistral-small-2603",
    "openrouter/anthropic/claude-sonnet-4.6",
    "openrouter/anthropic/claude-opus-4.7",
    # "openrouter/anthropic/claude-haiku-4.5"
]

def load_data(filepath='dataset.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def clean_answer(ans: str) -> str:
    if not isinstance(ans, str):
        return ""
    ans = re.sub(r'[^\w\s]', '', ans)
    ans = "".join(ans.split())
    return ans.upper()

SYSTEM_PROMPT = (
    "Jesteś precyzyjnym systemem rozwiązującym polskie krzyżówki. "
    "Otrzymasz definicję (pytanie) oraz maskę słowa, gdzie '_' oznacza nieznaną literę, "
    "a litery już podane to podpowiedź. "
    "Twoim zadaniem jest podać TYLKO I WYŁĄCZNIE ostateczne hasło. "
    "Nie dodawaj żadnego dodatkowego tekstu, żadnej interpunkcji, kropek ani zwrotów ugrzeczniających. "
    "Odpowiedź musi zawierać tylko odgadnięte polskie słowo."
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def call_llm(model: str, question: str, mask: str, expected_length: int, previous_fails: list) -> dict:
    user_prompt = f"Definicja: {question}\nLiczba liter: {expected_length}\nMaska: {mask}\n"
    if previous_fails:
        recent_fails = previous_fails[-5:]
        user_prompt += f"UWAGA! Te odpowiedzi na pewno są BŁĘDNE, wymyśl coś innego: {', '.join(recent_fails)}\n"
    user_prompt += "Podaj wprost nowe hasło:"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "timeout": 10,
    }

    if model.startswith("openrouter/"):
        kwargs["max_tokens"] = 1024
        kwargs["extra_body"] = {"reasoning": {"max_tokens": 1000}}

    t0 = time.time()
    response = completion(**kwargs)
    elapsed_ms = round((time.time() - t0) * 1000)

    msg = response.choices[0].message
    raw_text = msg.content
    usage = response.usage.to_dict() if response.usage else {}

    reasoning = getattr(msg, 'reasoning_content', None)
    thinking_blocks = getattr(msg, 'thinking_blocks', None)

    result = {
        "answer": clean_answer(raw_text),
        "raw_answer": raw_text,
        "messages": messages,
        "usage": usage,
        "elapsed_ms": elapsed_ms,
        "response_model": response.model,
    }

    if reasoning:
        result["reasoning"] = reasoning
    if thinking_blocks:
        result["thinking_blocks"] = [b.to_dict() if hasattr(b, 'to_dict') else b for b in thinking_blocks]

    return result

def generate_mask(answer: str, hints: int) -> str:
    length = len(answer)
    
    if hints >= length:
        return " ".join(list(answer))
        
    mask = list(answer[:hints]) + ["_"] * (length - hints)
    return " ".join(mask)

def model_log_path(model: str) -> str:
    slug = model.removeprefix("openrouter/").replace("/", "_")
    return os.path.join("logs", f"{slug}.json")

def load_existing_log(model: str) -> list:
    path = model_log_path(model)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def process_crosswords(dataset_path: str, models: list, output_path: str):
    data = load_data(dataset_path)
    results = []
    os.makedirs("logs", exist_ok=True)

    logs_by_model = {}
    for model in models:
        existing = load_existing_log(model)
        logs_by_model[model] = existing
        done = {e["question"] for e in existing}
        print(f"[{model}] Loaded {len(existing)} cached results, {len(done)} questions done")

    for count, row in enumerate(data, 1):
        question_text = row.get("pytanie", "")
        correct_answer = clean_answer(row.get("odpowiedz", ""))
        answer_length = len(correct_answer) if correct_answer else row.get("liter", 0)

        for model in models:
            done_questions = {e["question"] for e in logs_by_model[model]}
            if question_text in done_questions:
                cached = next(e for e in logs_by_model[model] if e["question"] == question_text)
                print(f"\n[{count}/{len(data)}] Model: {model} | Słowo: {correct_answer or '???'} ({answer_length} liter) | Pytanie: {question_text} [CACHED]")
                results.append({
                    "question": question_text,
                    "correct_answer": correct_answer,
                    "model": model,
                    "guessed": cached["guessed"],
                    "hints": cached["required_hints"],
                })
                continue

            print(f"\n[{count}/{len(data)}] Model: {model} | Słowo: {correct_answer or '???'} ({answer_length} liter) | Pytanie: {question_text}")
            guessed = False
            required_hints = None
            failed_attempts = []
            attempts = []
            skipped = False

            for hints in range(answer_length + 1):
                current_mask = generate_mask(correct_answer, hints)

                try:
                    llm_result = call_llm(model, question_text, current_mask, answer_length, failed_attempts)
                except Exception as e:
                    print(f"  [ERROR] {type(e).__name__}: {e}")
                    print(f"  [SKIP] Pomijam przypadek '{question_text}' dla modelu '{model}'")
                    skipped = True
                    break

                model_response = llm_result["answer"]

                attempt_entry = {
                    "hints": hints,
                    "mask": current_mask,
                    "raw_answer": llm_result["raw_answer"],
                    "cleaned_answer": model_response,
                    "correct": model_response == correct_answer,
                    "messages": llm_result["messages"],
                    "usage": llm_result["usage"],
                    "elapsed_ms": llm_result["elapsed_ms"],
                    "response_model": llm_result["response_model"],
                }
                if "reasoning" in llm_result:
                    attempt_entry["reasoning"] = llm_result["reasoning"]
                if "thinking_blocks" in llm_result:
                    attempt_entry["thinking_blocks"] = llm_result["thinking_blocks"]
                attempts.append(attempt_entry)

                if len(model_response) != answer_length:
                    print(f"  [MISS] '{model_response}' (wrong length: {len(model_response)} != {answer_length}, Mask: {current_mask})")
                    if model_response and model_response not in failed_attempts:
                        failed_attempts.append(model_response)
                    continue

                if model_response == correct_answer:
                    guessed = True
                    required_hints = hints
                    print(f"  [OK] '{model_response}' after {hints} hints (Mask: {current_mask})")
                    break
                else:
                    print(f"  [MISS] '{model_response}' (Mask: {current_mask})")
                    if model_response and model_response not in failed_attempts:
                        failed_attempts.append(model_response)

            if skipped:
                continue

            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "model": model,
                "guessed": guessed,
                "hints": required_hints
            })

            logs_by_model[model].append({
                "question": question_text,
                "correct_answer": correct_answer,
                "model": model,
                "guessed": guessed,
                "required_hints": required_hints,
                "attempts": attempts,
            })

            log_path = model_log_path(model)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(logs_by_model[model], f, ensure_ascii=False, indent=2)

    for model_name, entries in logs_by_model.items():
        log_path = model_log_path(model_name)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"Log saved: {log_path}")

    df = pd.DataFrame(results)
    df.to_csv("results_details.csv", index=False, encoding='utf-8')

    pivot_df = df.pivot(index='question', columns='model', values='hints')
    pivot_df = pivot_df.fillna("FAIL")

    correct_answers_series = df.drop_duplicates('question').set_index('question')['correct_answer']
    pivot_df.insert(0, 'Answer', correct_answers_series)
    pivot_df.to_csv(output_path, encoding='utf-8')

    print(f"CSV saved: results_details.csv, {output_path}")

    print("\n--- RESULTS SUMMARY ---")
    all_attempts = [a for entries in logs_by_model.values() for r in entries for a in r.get("attempts", [])]
    total_cost = sum(a.get("usage", {}).get("cost", 0) or 0 for a in all_attempts)
    total_tokens = sum(a.get("usage", {}).get("total_tokens", 0) or 0 for a in all_attempts)
    total_requests = len(all_attempts)
    print(f"Requests: {total_requests} | Tokens: {total_tokens} | Cost: ${total_cost:.6f}")

    for m in df['model'].unique():
        sub = df[df['model'] == m]
        success_count = sub['guessed'].sum()
        total = len(sub)
        accuracy = (success_count / total) * 100

        avg_hints = sub[sub['guessed'] == True]['hints'].mean()
        if pd.isna(avg_hints):
            avg_hints = 0.0

        print(f"{m:<50} | Accuracy: {success_count}/{total} ({accuracy:.0f}%) | Avg hints: {avg_hints:.2f}")

if __name__ == "__main__":
    process_crosswords("dataset.json", MODELS_TO_TEST, "results.csv")
