import json
import pandas as pd
import re
import os
import litellm
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

if not os.environ.get("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

litellm.suppress_debug_info = True

MODELS_TO_TEST = [
    "openrouter/openai/gpt-5.4-nano",
    "openrouter/google/gemma-4-31b-it",
    "openrouter/qwen/qwen3.5-flash-02-23",
    "openrouter/mistralai/mistral-small-2603"
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def call_llm(model: str, question: str, mask: str, previous_fails: list) -> str:
    system_prompt = (
        "Jesteś precyzyjnym systemem rozwiązującym polskie krzyżówki. "
        "Otrzymasz definicję (pytanie) oraz maskę słowa, gdzie '_' oznacza nieznaną literę, "
        "a litery już podane to podpowiedź. "
        "Twoim zadaniem jest podać TYLKO I WYŁĄCZNIE ostateczne hasło. "
        "Nie dodawaj żadnego dodatkowego tekstu, żadnej interpunkcji, kropek ani zwrotów ugrzeczniających. "
        "Odpowiedź musi zawierać tylko odgadnienięte polskie słowo."
    )
    
    user_prompt = f"Definicja: {question}\nMaska: {mask}\n"
    if previous_fails:
        recent_fails = previous_fails[-5:]
        user_prompt += f"UWAGA! Te odpowiedzi na pewno są BŁĘDNE, wymyśl coś innego: {', '.join(recent_fails)}\n"
    user_prompt += "Podaj wprost nowe hasło:"
    
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        timeout=15
    )
    res_text = response.choices[0].message.content
    return clean_answer(res_text)

def generate_mask(answer: str, hints: int) -> str:
    length = len(answer)
    
    if hints >= length:
        return " ".join(list(answer))
        
    mask = list(answer[:hints]) + ["_"] * (length - hints)
    return " ".join(mask)

def process_crosswords(dataset_path: str, models: list, output_path: str):
    data = load_data(dataset_path)
    results = []
    
    for count, row in enumerate(data, 1):
        question_text = row.get("pytanie", "")
        correct_answer = clean_answer(row.get("odpowiedz", ""))
        answer_length = len(correct_answer)
        
        print(f"\n--- Processing question {count}/{len(data)}: {question_text} (Answer: {correct_answer}) ---")
        
        for model in models:
            guessed = False
            required_hints = None
            failed_attempts = []
            
            for hints in range(answer_length + 1):
                current_mask = generate_mask(correct_answer, hints)
                
                try:
                    model_response = call_llm(model, question_text, current_mask, failed_attempts)
                except Exception as e:
                    print(f"[{model}] API/Rate Limit Error: {str(e)}")
                    model_response = ""
                
                if model_response == correct_answer:
                    guessed = True
                    required_hints = hints
                    print(f"  [OK] Model '{model}' GUESSED '{model_response}' after {hints} hints (Mask: {current_mask}).")
                    break
                else:
                    print(f"  [MISS] Model '{model}' answered: '{model_response}' (Mask: {current_mask})")
                    if model_response:
                        if model_response not in failed_attempts:
                            failed_attempts.append(model_response)
                    
            results.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "model": model,
                "guessed": guessed,
                "hints": required_hints
            })
            
    df = pd.DataFrame(results)
    
    df.to_csv("results_details.csv", index=False, encoding='utf-8')
    
    pivot_df = df.pivot(index='question', columns='model', values='hints')
    pivot_df = pivot_df.fillna("FAIL")
    
    correct_answers_series = df.drop_duplicates('question').set_index('question')['correct_answer']
    pivot_df.insert(0, 'Answer', correct_answers_series)
    
    pivot_df.to_csv(output_path, encoding='utf-8')
    
    print(f"\nDone! Saved detailed logs to 'results_details.csv'")
    print(f"Pivot table saved to: {output_path}")
    
    print("\n--- RESULTS SUMMARY ---")
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
