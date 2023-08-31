import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans




#Read the data
data = pd.read_csv("data.csv")
neighbors=pd.read_csv("NeighborhoodsUsa.csv")

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






#Claaing ny neighborhoods
ny_neighbors=neighbors[neighbors['city_name']=='New York']
list_ny_neighbors=ny_neighbors['neighborhood']
list_ny_neighbors=list(list_ny_neighbors)
list_ny_neighbors=[string.upper() for string in list_ny_neighbors]





# Load the English language model
nlp = spacy.load("en_core_web_sm")


# Clean and tokenize the text data, and convert it into phrases


phrases = []

for text in data['Text'][0:100]:
    doc = nlp(text)
    for sent in doc.sents:
        phrases.append(sent.text)



def filter_phrases_by_neighbors1(phrases, list_ny_neighbors):
    filtered_phrases = []
    corresponding_neighbors = []

    for phrase in phrases:
        phrase_upper = phrase.upper()  # Convert the phrase to uppercase
        for word in phrase_upper.split():  # Split the phrase into words
            if word in list_ny_neighbors:
                filtered_phrases.append(phrase)
                corresponding_neighbors.append(word)
                break  # Break the loop after finding the matching neighborhood word

    return filtered_phrases, corresponding_neighbors


def filter_phrases_by_neighbors(phrases, list_ny_neighbors):
    filtered_phrases = []
    corresponding_neighbors = []

    for phrase in phrases:
        for neighborhood in list_ny_neighbors:
            if neighborhood.upper() in phrase.upper():
                filtered_phrases.append(phrase)
                corresponding_neighbors.append(neighborhood)
                break  # Break the loop after finding the first matching neighborhood

    return filtered_phrases, corresponding_neighbors



filtered_phrases, corresponding_neighbors = filter_phrases_by_neighbors1(phrases, list_ny_neighbors)



####################CLustering######################

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(filtered_phrases)

# Perform Truncated SVD for dimensionality reduction
svd = TruncatedSVD(n_components=2)
X_2d = svd.fit_transform(X)

# Perform K-Means clustering
num_clusters = 3  # Change this to the number of clusters you want
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# Get cluster assignments for each phrase
cluster_labels = kmeans.labels_

# Create a scatter plot of the clusters
plt.figure(figsize=(10, 7))

for i in range(num_clusters):
    cluster_points = X_2d[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

plt.title('Cluster Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()





























# phrases = []

# for text in data['Text']:
#     doc = nlp(text)
#     sent_texts = [" ".join([token.text for token in sent]) for sent in doc.sents]
#     phrases.extend(sent_texts)


# adjacent_phrases = []

# for i, phrase in enumerate(phrases):
#     for j, word in enumerate(list_ny_neighbors):
#         if word in phrase.split():
#             if i > 0 and i < len(phrases) - 1:
#                 adjacent = " ".join([phrases[i-1], phrase, phrases[i+1]])
#                 adjacent_phrases.append((i, j, adjacent))

# df1 = pd.DataFrame(adjacent_phrases, columns=["PhraseIndex", "SearchWordIndex", "AdjacentPhrase"])


##Tenemos el caption de varios mercados. Como hablan de distintos vecindarios.
#Luego es descoponer lo que se esta diciendo, unsupevide are creating categories en si mismo












