import spacy

# Load the trained transformer-based model
nlp = spacy.load("custom_trf_model")

test_text = (
    "HQ has detected unusual Bulbasaur activity in the area. Field sensors logged anomalous behavior that suggests an imminent threat. "
    "Remember there are Pikachu and Charizard nearby — take care not to draw them into combat. You are to neutralize the bulbasaurs immediately. "
    "Report status once the target is down. Confirm mission status and any collateral damages."
)

doc = nlp(test_text)
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])