test_text = "HQ has detected unusual Bulbasaur activity in the area. Field sensors logged anomalous behavior that suggests an imminent threat. Remember there are Pikachu and Charizard nearby — take care not to draw them into combat. You are to neutralize the bulbasaurs immediately. Report status once the target is down. Confirm mission status and any collateral damages."
pokemon=["Pikachu","Charizard","Mewto","Bulbasaur"]
detected=[]
word=""
for text in test_text:
    if text==' ' or text==test_text[len(test_text)-1]:
        if word in pokemon:
            detected.append(word)
        word=""
        text=''
    word+=text
    
print(detected)
