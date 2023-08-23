import pandas as pd
import re
import spacy

#Read the data
data = pd.read_csv("data.csv")
neighbors=pd.read_csv("NeighborhoodsUsa.csv")

ny_neighbors=neighbors[neighbors['city_name']=='New York']
list_ny_neighbors=ny_neighbors['neighborhood']
list_ny_neighbors=list(list_ny_neighbors)

##Clean data and extract only the date in a string


def extract_time(text):
    time_pattern = r"\d{2}:\d{2}:\d{2} [APap][Mm]"
    match = re.search(time_pattern, text)

    if match:
        return match.group()
    else:
        return None


def extract_date(text):
    date_pattern = r"\d{1,2}/\d{1,2}/\d{4}"  # MM/DD/YYYY format
    match = re.search(date_pattern, text)

    if match:
        return match.group()
    else:
        return None


data["Time"] = data["Text"].apply(extract_time)
data["Date"] = data["Text"].apply(extract_date)


# Load the English language model
nlp = spacy.load("en_core_web_sm")

text1=data['Text'][0]
doc=nlp(text1)

phrases = []

for sent in doc.sents:
    sent_phrases = [token.text for token in sent]
    phrases.append(" ".join(sent_phrases))


list_ny_neighbors=[string.upper() for string in list_ny_neighbors]


# adjacent_phrases = []

# for i, phrase in enumerate(phrases):
#     for j, word in enumerate(list_ny_neighbors):
#         if word in phrase:
#             adjacent = " ".join([phrases[i-1], phrase, phrases[i+1]])
#             adjacent_phrases.append((i, j, adjacent))

# df1 = pd.DataFrame(adjacent_phrases, columns=["PhraseIndex", "SearchWordIndex", "AdjacentPhrase"])


adjacent_phrases = []

for i, phrase in enumerate(phrases):
    for j, word in enumerate(list_ny_neighbors):
        if word in phrase.split():
            if i > 0 and i < len(phrases) - 1:
                adjacent = " ".join([phrases[i-1], phrase, phrases[i+1]])
                adjacent_phrases.append((i, j, adjacent))

df1 = pd.DataFrame(adjacent_phrases, columns=["PhraseIndex", "SearchWordIndex", "AdjacentPhrase"])

















