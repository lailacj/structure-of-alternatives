# A simple script that puts all the stories and prompts into a data frame. 
# " [MASk] " is included in the prompts. 
# These prompts are made for BERT usage.

import pandas as pd

# Story and BERT promppt
data = [
    {"story": "bag", 
     "prompt": "You and your friend Taylor are sitting in a park. You pull out a piece of paper and tell Taylor, 'I just had an idea that I want to remember, but don't have anything to write with.' Taylor looks through her handbag and responds, 'I only have [MASK].'"},
    
    {"story": "bakery", 
     "prompt": "You and your friend Chris walk by a bakery. You say, 'I want some dessert.' Chris looks inside the bakery and responds, 'They only have [MASK].'"},
    
    {"story": "beach", 
     "prompt": "You and your friend Jess are packing to go on vacation. You tell Jess, 'There's a nice beach near our hotel, make sure you bring something to wear.' Jess looks in the closet and responds, 'I only have [MASK].'"},

    {"story": "cold", 
     "prompt": "You and your friend Kennedy rent a small cabin in the mountains for a weekend. When you arrive, you shiver and say 'I'm freezing!' Kennedy looks around the cabin and responds, 'They only have [MASK].'"},

    {"story": "cut", 
     "prompt": "You and your friend Blair come back from a long day shopping. You want to try on your new sweater, and you tell Blair, 'I want to remove this tag.' Blair looks through some drawers and responds, 'I only have [MASK].'"},

    {"story": "fridge", 
     "prompt": "You and your friend Sam go for a long walk together. After the walk, you go back to Sam's house. You say to Sam, 'I'm thirsty.' Sam opens the fridge and responds, 'I only have [MASK].'"},

    {"story": "gym", 
     "prompt": "You and your friend Lee are registering for classes at a fitness center. You say to Lee, 'Let's sign up for some team sports.' Lee looks at the list of classes and responds, 'They only have [MASK].'"},

    {"story": "hot", 
     "prompt": "You and your friend Grayson visit his grandparents' lake house after hiking on a hot summer day. You tell Grayson, 'It's sweltering in here!' Grayson talks to his grandparents and responds, 'They only have [MASK].'"},

    {"story": "mall", 
     "prompt": "You and your friend Alex are hanging out in a newly opened shopping mall. You say to Alex, 'Let's get lunch!' Alex looks at the map of the mall and responds, 'This mall only has [MASK].'"},

    {"story": "mask", 
     "prompt": "You and your friend Carly go grocery shopping together. As you're about to enter the supermarket, you reach into your pocket and realize, 'I forgot to bring my mask!' Carly looks through her bag and responds, 'I only have [MASK].'"},

    {"story": "meat", 
     "prompt": "Your friend MJ is calling a ramen restaurant to order takeout. You tell MJ, 'I'd like meat in mine, please.' MJ gives your order over the phone, and then responds, 'They only have [MASK].'"},

    {"story": "restaurant", 
     "prompt": "You and your friend Billie go to a new restaurant. While you're there, you find out that it's Billie's 21st birthday. You say to Billie, 'Let's order some drinks!' Billie looks at the menu and responds, 'They only have [MASK].'"},
    
    {"story": "salad", 
     "prompt": "You and your friend Ali are preparing dinner together. You tell Ali, 'I want to make a salad!' Ali runs to the corner store, and texts you, 'They only have [MASK].'"},

    {"story": "science", 
     "prompt": "You and your friend Drew are looking for books in a library. You say, 'I want to learn more about science.' Drew looks at the library catalogue and responds, 'They only have books about [MASK].'"},

    {"story": "throw", 
     "prompt": "You and your friend Jo are relaxing at Jo's house.You tell Jo, 'It's sunny outside, let's play catch.' Jo looks through the garage and responds, 'We only have [MASK].'"},

    {"story": "transport", 
     "prompt": "You and your friend Dylan have been studying together at his uncle's house. Dylan suggests going into town to take a break. You say, 'Sure, but I don't want to walk.' Dylan looks in the garage and responds, 'He only has [MASK].'"}
]

# Create the DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("../data/prompts_BERT.csv", index=False)