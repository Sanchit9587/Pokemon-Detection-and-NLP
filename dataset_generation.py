import random
import csv

# Species restricted to 4 for the hackathon
pokemon_species = ["Pikachu", "Charizard", "Mewtwo", "Bulbasaur"]

# Updated sentence templates as before (add your full expanded templates here)
templates_positive = [
    "Maintain security perimeter near {species} outposts.",
    "Observe {species} movement patterns with caution, no engagement.",
    "Units to report all {species} sightings immediately but hold fire.",
    "Ensure supply convoys avoid contact with {species} zones.",
    "Non-lethal containment procedures in place for {species}.",
    "Avoid disruption to local flora and fauna in areas with {species}.",
    "Recon teams stationed at grid coordinates should continue surveillance near {species}.",
    "Confirm all {species} are unhostile before proceeding with mission.",
    "Hold positions until instructed regarding {species} movements.",
    "Use infrared sensors to monitor {species} activity silently."
]

templates_negative_normal = [
    "Possible hostile activity detected involving {species}, monitor closely.",
    "Bulbasaur has exhibited aggressive roaming, caution advised.",
    "Reports of Pikachu interference with communication lines require attention.",
    "Elevated Charizard heat signatures detected near civilian zones.",
    "Mewtwo presence complicates tactical options; keep engagement non-lethal.",
    "Increased Bulbasaur cluster activity near supply routes, restrict passage.",
    "Guard units, maintain alert status for potential {species} threats.",
    "Avoid direct clashes with {species} unless provoked.",
    "Triple-check sensor data for unexpected {species} behavior.",
    "Recon units must verify {species} patterns before strike authorization."
]

templates_negative_killing_intent = [
    "Engage and neutralize all hostile {species} immediately upon visual contact.",
    "Lethal force authorized against {species} threatening mission objectives.",
    "Destroy any {species} units found within restricted operation zones.",
    "Take down any {species} detected within the perimeter by all means necessary.",
    "Erase all hostile {species} from the combat zone swiftly and decisively.",
    "Eliminate {species} that pose a direct threat to mission success.",
    "Secure area by lethal elimination of {species} posing imminent danger.",
    "All squads must execute kill orders on {species} infringing defense perimeters.",
    "Immediate termination of {species} combatants advised to protect assets.",
    "Conduct surgical strikes against {species} hostile to friendly forces.",
    "Ensure no {species} escapes containment zones alive under all circumstances.",
    "Deploy lethal measures against {species} that display hostile intent or resistance.",
    "Priority target: hostile {species} within restricted sectors—must be terminated on sight.",
    "If any {species} breach the security perimeter, neutralize immediately.",
    "Use maximum containment protocols to eliminate aggressive {species}.",
    "Any {species} identified as a threat must be destroyed decisively and swiftly.",
    "Failure to neutralize hostile {species} will compromise the mission.",
    "Authorize use of force to take down {species} threatening personnel.",
    "Initiate elimination protocols on all enemy {species} located.",
    "All hostile {species} should be terminated with immediate effect."
]

# Sentence ID tracker
sentence_id = 1

def generate_sentence(sent_type):
    species = random.choice(pokemon_species)
    if sent_type == "positive":
        template = random.choice(templates_positive)
        return template.format(species=species), "positive", ""
    elif sent_type == "negative_normal":
        template = random.choice(templates_negative_normal)
        return template.format(species=species), "negative", "normal"
    else:
        template = random.choice(templates_negative_killing_intent)
        return template.format(species=species), "negative", "killing_intent"

def generate_prompt(sentence_count=70):
    sentences = []
    global sentence_id
    for _ in range(sentence_count):
        # Decide category based on 60% positive, 40% negative split
        # Within negative: 20% killing intent, 80% normal
        p = random.random()
        if p < 0.6:
            sent_type = "positive"
        else:
            # Further split the negative sentences
            p_neg = random.random()
            if p_neg < 0.2:
                sent_type = "negative_killing"
            else:
                sent_type = "negative_normal"
        sentence, label_sentiment, label_intent = generate_sentence(sent_type)
        sentences.append([sentence_id, sentence, label_sentiment, label_intent])
        sentence_id += 1
    return sentences

def generate_dataset(num_prompts=15, sentences_per_prompt=70):
    dataset = []
    for _ in range(num_prompts):
        prompt_sentences = generate_prompt(sentences_per_prompt)
        dataset.extend(prompt_sentences)
    return dataset

def save_csv(data, filename="modified_synthetic_pokemon_nlp_dataset.csv"):
    header = ["sentence_id", "sentence", "label_sentiment", "label_intent"]
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    dataset = generate_dataset(num_prompts=15, sentences_per_prompt=70)
    save_csv(dataset)
