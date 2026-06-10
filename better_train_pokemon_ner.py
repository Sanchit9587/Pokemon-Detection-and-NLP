import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import os

# Use transformer model for better accuracy
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("Downloading en_core_web_trf model...")
    from spacy.cli import download
    download("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf")

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add label
ner.add_label("POKEMON")

# Pokémon names and variants
pokemon_names = [
    "Pikachu", "Charizad", "Bulbasaur", "Mewto",
    "pikachu", "charizad", "bulbasaur", "mewto",
    "Pikachus", "Charizads", "Bulbasaurs", "Mewtos",
    "Pikachu's", "Charizad's", "Bulbasaur's", "Mewto's"
]

# Data augmentation: generate sentences
def generate_sentences():
    templates = [
        "I saw {name} in the forest.",
        "{name} was playing with children.",
        "The trainer called for {name}.",
        "A group of {name} appeared suddenly.",
        "No Pokémon were seen today.",
        "The equipment was damaged overnight.",
        "{name} tracks were found near the river.",
        "We heard rumors about {name}'s abilities.",
        "There are reports of {name} causing trouble.",
        "The scientist observed {name} closely.",
        "Nothing unusual happened in the lab.",
        "The city was quiet last night.",
        "{name} and {name2} worked together on the mission.",
        "The battle was won by {name}.",
        "No signs of Pikachu, Charizad, Bulbasaur, or Mewto."
    ]
    data = []
    for name in pokemon_names:
        # Singular/plural/possessive
        if name.lower() in ["pikachu", "charizad", "bulbasaur", "mewto"]:
            # Add context with another Pokémon
            for other in pokemon_names:
                if other != name and other.lower() in ["pikachu", "charizad", "bulbasaur", "mewto"]:
                    s = f"{name} and {other} were spotted near the lake."
                    data.append((s, {"entities": [
                        (0, len(name), "POKEMON"),
                        (len(name) + 5, len(name) + 5 + len(other), "POKEMON")
                    ]}))
        for template in templates:
            if "{name2}" in template:
                # Pick another Pokémon for {name2}
                for other in pokemon_names:
                    if other != name:
                        s = template.format(name=name, name2=other)
                        ents = []
                        idx1 = s.find(name)
                        if idx1 != -1:
                            ents.append((idx1, idx1 + len(name), "POKEMON"))
                        idx2 = s.find(other, idx1 + len(name))
                        if idx2 != -1:
                            ents.append((idx2, idx2 + len(other), "POKEMON"))
                        if ents:
                            data.append((s, {"entities": ents}))
            else:
                s = template.format(name=name)
                idx = s.find(name)
                if idx != -1 and "No Pokémon" not in s and "Nothing unusual" not in s and "quiet" not in s:
                    data.append((s, {"entities": [(idx, idx + len(name), "POKEMON")]}))
                else:
                    # Negative examples (no entities)
                    data.append((s, {"entities": []}))
    return data

TRAIN_DATA = generate_sentences()

# Shuffle and split for training
random.shuffle(TRAIN_DATA)

# Training loop
optimizer = nlp.resume_training()
n_iter = 30  # More iterations for better learning

for itn in range(n_iter):
    random.shuffle(TRAIN_DATA)
    losses = {}
    # Use minibatch for efficiency
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.5))
    for batch in batches:
        examples = []
        for text, annots in batch:
            examples.append(Example.from_dict(nlp.make_doc(text), annots))
        nlp.update(examples, drop=0.3, sgd=optimizer, losses=losses)
    print(f"Iteration {itn+1}, Losses: {losses}")

# Save model
output_dir = "custom_trf_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")