from openie import StanfordOpenIE
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
import os, sys
roberta = TransformerDocumentEmbeddings('roberta-base') 
#bert-large-uncased-whole-word-masking-finetuned-squad  roberta-base all-MiniLM-L6-v2
#kw_model = KeyBERT(model=roberta)
doc = "Did the accident happen when the involved vehicles were speeding?" #Was the main cause of the accident due to speeding vehicles?
kw_model = KeyBERT()
#keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 10), stop_words=None,  use_mmr=True, diversity=0.7)
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 4), stop_words=None, use_mmr=True, diversity=1,top_n=3)
print(keywords)

properties = {
    'openie.affinity_probability_cap': 1/3,
}

with StanfordOpenIE(properties=properties) as client:
    #text = 'the accident still happen if there were fewer vehicles on road'
    #text ='Was the main cause of the accident due to speeding vehicles?'
    text = ''.join(keywords[0][0])
    print('Text: %s.' % text)
    for triple in client.annotate(text):
        print('|-', triple)

    #graph_image = 'graph.png'
    #client.generate_graphviz_graph(text, graph_image)
    #print('Graph generated: %s.' % graph_image)