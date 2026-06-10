from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import json

# Example NER output and prompt
ner_species = ["Pikachu", "Charizard", "Bulbasaur"]
military_prompt = """
Alpha team encountered multiple entities in sector 7G. Pikachu units displayed aggressive electrical discharges, Charizard provided aerial cover with incendiary attacks, Bulbasaur clusters remained stationary and did not engage.
"""

# Compose the prompt for the LLM
def build_llm_prompt(entities, original_prompt):
    species_list = ', '.join(entities)
    return (
        f"You are a military analyst. Given the following battlefield report:\n"
        f"{original_prompt}\n\n"
        f"Analyze the following Pokémon species: {species_list}.\n"
        f"For each, determine if it is 'hostile' or 'to be protected' based on the report. "
        f"Output a JSON object with two lists: 'hostile' and 'non_hostile'."
    )

llm_prompt = build_llm_prompt(ner_species, military_prompt)

# Load a small LLM (e.g., distilgpt2)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate the output
outputs = generator(llm_prompt, max_length=512, num_return_sequences=1)
response = outputs[0]['generated_text']

# Extract JSON from the response (simple heuristic)
import re

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

json_output = extract_json(response)
print(json.dumps(json_output, indent=2))