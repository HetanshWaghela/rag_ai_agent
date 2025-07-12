from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
df=pd.read_csv("movie_reviews_dataset.csv")
embedding=OllamaEmbeddings(model="mxbai-embed-large")

db_path= "./chroma_db"
add_doc= not os.path.exists(db_path)



if add_doc:
    documents=[]
    ids=[]
    for i,row in df.iterrows():
        document=Document(
            page_content= row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)

        )
        ids.append(str(i))
        documents.append(document)

vector_store=Chroma(
    collection_name="movie_reviews",
    persist_directory=db_path,
    embedding_function=embedding
)
if add_doc:
    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()

retriever=vector_store.as_retriever(
    search_kwargs={"k": 5},
)
