from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os



def compute_bert_text_embeddings(BASE):

    data_file = "ad-tweet-data.txt"
    data = open(os.path.join(BASE, data_file), "r").readlines()
    print(len(data))
    text_embeddings_all = []
    '''
    # https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
    Base Model: Teacher: mUSE; Student: distilbert-base-multilingual
    Max Sequence Length: 128
    Dimensions: 512
    Normalized Embeddings: false
    Training Data: Multi-Lingual model of Universal Sentence Encoder for 50 languages.
    '''
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    for idx in range(len(data)):
        # print(data[idx])
        line = json.loads(data[idx])
        text_content = line["ad_text"] # str
        translated_text_content = line["ad_text_translated"] #str
        text_embedding = model.encode(text_content)
        text_embeddings_all.append(text_embedding)
        print(idx, text_embedding.shape)
    
    text_embeddings_all = np.vstack(text_embeddings_all)
    print(text_embeddings_all.shape) # (x, 512)

    np.savez(f"{BASE}/bert-embeddings.npz", text=text_embeddings_all)
    
    return



def main():

    bert_base_dir = "/INET/socialnets3/static00/yvekaria/bert_data"
    
    compute_bert_text_embeddings(bert_base_dir)

    return
            

if __name__=="__main__":
    
    main()
