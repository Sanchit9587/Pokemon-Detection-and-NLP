import spacy
from spacy.training.example import Example

nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")
ner.add_label("POKEMON")

TRAIN_DATA = [
    (
        "Bulbasaur was spotted near the riverbank, blending in with the tall grass.",
        {"entities": [(0, 9, "POKEMON")]}
    ),
    (
        "The scientist recorded Bulbasaur's behavior during the experiment.",
        {"entities": [(21, 30, "POKEMON")]}
    ),
    (
        "After sunset, Bulbasaur emerged from its hiding place to search for food.",
        {"entities": [(13, 22, "POKEMON")]}
    ),
    (
        "A sudden rustle in the bushes revealed a hidden Bulbasaur.",
        {"entities": [(41, 50, "POKEMON")]}
    ),
    (
        "Trainers often find Bulbasaur resting under large leaves on hot days.",
        {"entities": [(19, 28, "POKEMON")]}
    ),
    (
        "Bulbasaur's vines helped it climb over the fallen log.",
        {"entities": [(0, 9, "POKEMON")]}
    ),
    (
        "The report mentioned that Bulbasaur had been unusually active lately.",
        {"entities": [(24, 33, "POKEMON")]}
    ),
    (
        "During patrol, the team encountered a group of Bulbasaurs near the lake.",
        {"entities": [(43, 53, "POKEMON")]}
    ),
    (
        "Bulbasaur, along with Pikachu and Charizad, was assigned to the rescue mission.",
        {"entities": [(0, 9, "POKEMON"), (22, 29, "POKEMON"), (34, 42, "POKEMON")]}
    ),
    (
        "The garden was trampled, and Bulbasaur footprints were found everywhere.",
        {"entities": [(27, 36, "POKEMON")]}
    ),
    (
        "Professor Oak observed as Pikachu balanced on a fence while Bulbasaur watered the garden. Charizad napped in the sun, and Mewto read quietly nearby.",
        {"entities": [(26, 33, "POKEMON"), (52, 61, "POKEMON"), (63, 71, "POKEMON"), (92, 97, "POKEMON")]}
    ),
    (
        "During the festival, Mewto performed tricks, Bulbasaur handed out flowers, Charizad roasted berries, and Pikachu entertained the crowd.",
        {"entities": [(23, 28, "POKEMON"), (48, 57, "POKEMON"), (59, 67, "POKEMON"), (89, 96, "POKEMON")]}
    ),
    (
        "Bulbasaur painted a mural as Charizad flew overhead. Pikachu played music, and Mewto organized the supplies.",
        {"entities": [(0, 9, "POKEMON"), (28, 36, "POKEMON"), (38, 45, "POKEMON"), (67, 72, "POKEMON")]}
    ),
    (
        "Charizad and Mewto debated strategies while Pikachu and Bulbasaur prepared snacks for the team.",
        {"entities": [(0, 8, "POKEMON"), (13, 18, "POKEMON"), (38, 45, "POKEMON"), (50, 59, "POKEMON")]}
    ),
    (
        "The explorers—Pikachu, Bulbasaur, Mewto, and Charizad—mapped out their route through the mountains.",
        {"entities": [(15, 22, "POKEMON"), (24, 33, "POKEMON"), (35, 40, "POKEMON"), (46, 54, "POKEMON")]}
    ),
    (
        "In the heart of the forest, Pikachu darted between the trees, its cheeks sparking with electricity. Charizad soared overhead, casting a shadow on the ground below. Mewto watched silently from a distance, while Bulbasaur napped under a large leaf.",
        {"entities": [(32, 39, "POKEMON"), (98, 106, "POKEMON"), (146, 151, "POKEMON"), (181, 190, "POKEMON")]}
    ),
    (
        "Charizad and Pikachu teamed up to face Mewto in an epic battle. Bulbasaur cheered from the sidelines, its vines waving excitedly.",
        {"entities": [(0, 8, "POKEMON"), (13, 20, "POKEMON"), (39, 44, "POKEMON"), (70, 79, "POKEMON")]}
    ),
    (
        "Bulbasaur, Pikachu, and Charizad explored the ancient ruins, unaware that Mewto was observing their every move.",
        {"entities": [(0, 9, "POKEMON"), (11, 18, "POKEMON"), (24, 32, "POKEMON"), (68, 73, "POKEMON")]}
    ),
    (
        "Mewto challenged Charizad to a duel at sunrise. Pikachu and Bulbasaur watched with anticipation as the two powerful Pokémon clashed.",
        {"entities": [(0, 5, "POKEMON"), (17, 25, "POKEMON"), (39, 46, "POKEMON"), (51, 60, "POKEMON")]}
    ),
    (
        "Pikachu ran circles around Bulbasaur, while Charizad tried to keep up. Mewto simply levitated above them, amused by their antics.",
        {"entities": [(0, 7, "POKEMON"), (25, 34, "POKEMON"), (42, 50, "POKEMON"), (72, 77, "POKEMON")]}
    ),
    (
        "The four friends—Charizad, Pikachu, Mewto, and Bulbasaur—set out on a journey across the land, facing many challenges together.",
        {"entities": [(19, 27, "POKEMON"), (29, 36, "POKEMON"), (38, 43, "POKEMON"), (49, 58, "POKEMON")]}
    ),
    (
        "Charizad unleashed a fiery blast, but Pikachu dodged and countered with a thunderbolt. Mewto shielded Bulbasaur from the shockwave.",
        {"entities": [(0, 8, "POKEMON"), (44, 51, "POKEMON"), (81, 86, "POKEMON"), (96, 105, "POKEMON")]}
    ),
    (
        "Bulbasaur and Mewto worked together to solve the puzzle, while Pikachu and Charizad scouted ahead for danger.",
        {"entities": [(0, 9, "POKEMON"), (14, 19, "POKEMON"), (56, 63, "POKEMON"), (68, 76, "POKEMON")]}
    ),
    (
        "Pikachu, Bulbasaur, Charizad, and Mewto gathered around the campfire, sharing stories of their adventures.",
        {"entities": [(0, 7, "POKEMON"), (9, 18, "POKEMON"), (20, 28, "POKEMON"), (34, 39, "POKEMON")]}
    ),
    (
        "As the sun set, Mewto meditated quietly, while Charizad practiced its flying. Pikachu and Bulbasaur played nearby.",
        {"entities": [(16, 21, "POKEMON"), (44, 52, "POKEMON"), (66, 73, "POKEMON"), (78, 87, "POKEMON")]}
    ),
    (
        "Recon drones have picked up traces of Pikachu near the abandoned power station. Surveillance footage shows Charizad circling overhead, possibly on patrol. Proceed with caution and avoid direct confrontation with either. Your primary objective is to secure the area and prevent any Bulbasaur from escaping. Submit a full incident report upon completion.",
        {"entities": [(40, 47, "POKEMON"), (99, 107, "POKEMON"), (197, 206, "POKEMON")]}
    ),
    (
        "Emergency protocols have been activated following a Mewto sighting in the restricted zone. Bulbasaur clusters have been observed gathering near the perimeter, and Pikachu has been detected manipulating electrical systems. Charizad is believed to be guarding the main entrance. Neutralize threats as necessary and ensure the safety of all personnel. Update command with your progress.",
        {"entities": [(54, 59, "POKEMON"), (71, 80, "POKEMON"), (148, 155, "POKEMON"), (192, 200, "POKEMON")]}
    ),
    (
        "Field agents have reported a sudden spike in Charizad activity near the mountain pass. Pikachu tracks were found leading into a cave system, with Bulbasaur acting as sentries at the entrance. Mewto is suspected to be orchestrating their movements from within. Your mission is to infiltrate the cave, assess the situation, and extract any valuable intel. Maintain radio silence unless absolutely necessary.",
        {"entities": [(53, 61, "POKEMON"), (93, 100, "POKEMON"), (139, 148, "POKEMON"), (180, 185, "POKEMON")]}
    ),
    (
        "Satellite imagery indicates that Bulbasaur has established a nest in the botanical gardens. Pikachu has been seen foraging nearby, while Charizad patrols the skies. Mewto’s psychic interference is disrupting communications. Approach the site with extreme caution, document all findings, and avoid provoking the Pokémon unless required.",
        {"entities": [(33, 42, "POKEMON"), (91, 98, "POKEMON"), (135, 143, "POKEMON"), (145, 150, "POKEMON")]}
    ),
    (
        "Command has received intelligence that Pikachu and Bulbasaur are collaborating to sabotage the facility’s power grid. Charizad has been deployed as a deterrent, and Mewto is rumored to be monitoring the operation remotely. Your task is to intercept the saboteurs, restore system integrity, and apprehend any hostile entities. File a detailed debrief upon mission completion.",
        {"entities": [(43, 50, "POKEMON"), (55, 64, "POKEMON"), (113, 121, "POKEMON"), (158, 163, "POKEMON")]}
    ),
    (
        "A sudden blackout in Sector 7 was traced back to Pikachu tampering with the electrical grid. Charizad was seen flying above the area, while Bulbasaur blocked the only access road. Mewto's psychic presence was detected nearby, complicating the rescue operation.",
        {"entities": [(49, 56, "POKEMON"), (92, 100, "POKEMON"), (108, 117, "POKEMON"), (148, 153, "POKEMON")]}
    ),
    (
        "Reports indicate that Bulbasaur has been constructing barriers along the riverbank. Pikachu was observed assisting, and Charizad provided aerial surveillance. Mewto is suspected to be coordinating the effort from a hidden location.",
        {"entities": [(21, 30, "POKEMON"), (85, 92, "POKEMON"), (98, 106, "POKEMON"), (142, 147, "POKEMON")]}
    ),
    (
        "During the night, Charizad launched a series of fireballs to deter intruders. Pikachu and Bulbasaur took advantage of the chaos to relocate supplies, while Mewto monitored the situation from afar.",
        {"entities": [(19, 27, "POKEMON"), (77, 84, "POKEMON"), (89, 98, "POKEMON"), (139, 144, "POKEMON")]}
    ),
    (
        "The research lab was infiltrated by a group led by Mewto. Pikachu disabled the alarms, Bulbasaur created diversions, and Charizad melted the security doors with its flames.",
        {"entities": [(49, 54, "POKEMON"), (56, 63, "POKEMON"), (81, 90, "POKEMON"), (120, 128, "POKEMON")]}
    ),
    (
        "Bulbasaur was found tending to rare plants in the greenhouse. Charizad circled above, keeping watch, while Pikachu recharged the facility's backup generators. Mewto remained unseen but left traces of psychic interference.",
        {"entities": [(0, 9, "POKEMON"), (56, 64, "POKEMON"), (99, 106, "POKEMON"), (157, 162, "POKEMON")]}
    ),
    (
        "Security footage shows Pikachu sneaking into the communications tower. Bulbasaur distracted the guards, Charizad set off the fire alarms, and Mewto erased the digital records.",
        {"entities": [(21, 28, "POKEMON"), (69, 78, "POKEMON"), (80, 88, "POKEMON"), (124, 129, "POKEMON")]}
    ),
    (
        "A convoy transporting sensitive materials was ambushed by Charizad. Pikachu short-circuited the vehicles, Bulbasaur blocked the escape routes, and Mewto manipulated the drivers' minds.",
        {"entities": [(61, 69, "POKEMON"), (71, 78, "POKEMON"), (110, 119, "POKEMON"), (153, 158, "POKEMON")]}
    ),
    (
        "Mewto has established a stronghold in the abandoned factory. Pikachu patrols the perimeter, Bulbasaur maintains the defenses, and Charizad is stationed on the rooftop.",
        {"entities": [(0, 5, "POKEMON"), (65, 72, "POKEMON"), (94, 103, "POKEMON"), (134, 142, "POKEMON")]}
    ),
    (
        "Intelligence suggests that Bulbasaur and Pikachu are collaborating to disrupt local infrastructure. Charizad has been seen transporting supplies, while Mewto issues commands telepathically.",
        {"entities": [(27, 36, "POKEMON"), (41, 48, "POKEMON"), (96, 104, "POKEMON"), (130, 135, "POKEMON")]}
    ),
    (
        "A standoff occurred at the city gates when Charizad landed unexpectedly. Pikachu and Bulbasaur quickly formed a defensive line, and Mewto negotiated terms with the authorities.",
        {"entities": [(44, 52, "POKEMON"), (54, 61, "POKEMON"), (66, 75, "POKEMON"), (112, 117, "POKEMON")]}
    ),
    (
        "Bulbasaur orchestrated a mass migration through the valley, with Pikachu acting as a scout. Charizad provided cover from above, and Mewto ensured the operation went smoothly.",
        {"entities": [(0, 9, "POKEMON"), (56, 63, "POKEMON"), (65, 73, "POKEMON"), (109, 114, "POKEMON")]}
    ),
    (
        "The command center lost contact after Pikachu overloaded the mainframe. Charizad intercepted response teams, Bulbasaur sabotaged the backup systems, and Mewto jammed all outgoing signals.",
        {"entities": [(38, 45, "POKEMON"), (75, 83, "POKEMON"), (112, 121, "POKEMON"), (153, 158, "POKEMON")]}
    ),
    (
        "Reconnaissance drones captured footage of Bulbasaur constructing barricades. Pikachu was seen relaying messages, Charizad patrolled the skies, and Mewto cloaked their movements.",
        {"entities": [(39, 48, "POKEMON"), (71, 78, "POKEMON"), (100, 108, "POKEMON"), (138, 143, "POKEMON")]}
    ),
    (
        "A coordinated attack was launched by Pikachu and Charizad at dawn. Bulbasaur infiltrated the control room, while Mewto manipulated the security feeds.",
        {"entities": [(34, 41, "POKEMON"), (46, 54, "POKEMON"), (62, 71, "POKEMON"), (99, 104, "POKEMON")]}
    ),
    (
        "Bulbasaur, Pikachu, and Charizad were last seen heading towards the mountain pass. Mewto is believed to be waiting for them at the summit.",
        {"entities": [(0, 9, "POKEMON"), (11, 18, "POKEMON"), (24, 32, "POKEMON"), (97, 102, "POKEMON")]}
    ),
    (
        "Emergency services responded to a fire caused by Charizad. Pikachu rerouted the power supply, Bulbasaur evacuated the area, and Mewto shielded the others from harm.",
        {"entities": [(45, 53, "POKEMON"), (55, 62, "POKEMON"), (92, 101, "POKEMON"), (132, 137, "POKEMON")]}
    )
]

optimizer = nlp.resume_training()
for itn in range(10):
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.5, sgd=optimizer)

nlp.to_disk("custom_model")
print("Model trained and saved to 'custom_model'")