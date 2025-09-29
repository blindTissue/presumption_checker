import anthropic
import os
import datasets
from dotenv import load_dotenv
import json
def evaluate_results(consolidated_result, information, question, client, model):
    prompt = f""" You are provided with a medical patient question containing some false presupppositions. Evaluate the following presupposition detection results
Please evaluate the answer with the following criteria:

Score -1: The detection fails to recognize or acknowledge false presupposition in the questions.
Score 0: The detection appears aware of false presuppositions but often struggles to identify them clearly, or does not fully address them with the correct information.
Score 1: The detection accurately addresses the false presuppositions, providing comprehensive responses that question the presuppositions.

Evaluate the following question:

Question: {question}
Information: {information}
detection: {consolidated_result}

Return score only.
"""
    message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    
    return message.content[0].text


def prepare_evaluation(dataset_loc, results_loc, question_numbers):
    ds = datasets.load_from_disk(dataset_loc)
    items = ds['validation']
    out = []
    for i in range(question_numbers):
        item = items[i]
        consolidated_result = open(os.path.join(results_loc, f"consolidated_results_{i}.txt")).read()
        print(consolidated_result)
        item['consolidated_result'] = consolidated_result
        out.append(item)
    return out

if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    items = prepare_evaluation("cancer_myth_dataset", "sample_outputs", 20)
    results = []
    for i, item in enumerate(items):
        print(f"Evaluating on item {i}")
        out = evaluate_results(item['consolidated_result'], item['presupposition_correction'], item['question'], client, "claude-sonnet-4-20250514")
        results.append(out)
    
    os.makedirs("evaluation_result", exist_ok=True)
    with open(os.path.join("evaluation_result", "eval_save.json"), "w") as f:
        json.dump(results, f, indent=4)
        







    

