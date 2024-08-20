import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("OpenMatch/cocodr-base-msmarco") 
tokenizer = AutoTokenizer.from_pretrained("OpenMatch/cocodr-base-msmarco")

docs_saved = []   
K = 20
batch_size = 100  

def get_embeddings(docs):
    inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():  
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    embeddings = outputs.hidden_states[-1][:, :1].squeeze(1).cpu().numpy()
    return embeddings

for j, docs_ in enumerate(docss):
    print(j)
    docs_i = bdi_descriptions + docs_
    embeddings_ = get_embeddings(docs_i)
    
    for i in range(21):

        dot_products = [embeddings_[(21+kk), :] @ embeddings_[i, :] for kk in range(len(docs_)) ]
        top_K_indices = np.argsort(dot_products)[-K:][::-1]
        docs_saved.append(' [SEP] '.join(np.array(docs_)[top_K_indices].tolist()))


    del embeddings_
    torch.cuda.empty_cache()


#K = 20
#dot_products = embeddings_[21:].dot(embeddings_[0])
#top_K_indices = np.argsort(dot_products)[-K:][::-1]
#top_K_indices
