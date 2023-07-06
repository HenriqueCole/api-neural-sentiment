import spacy

# Load the spacy model
nlp = spacy.load("pt_core_news_sm")


def similiarText(text1, text2):
    # Create a doc object for each text
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # Return the similiarity between the two texts
    return doc1.similarity(doc2) * 100
