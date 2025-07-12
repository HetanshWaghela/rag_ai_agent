from langchain_ollama import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate
from vdb import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a helpful assistant. You can expertly answer questions about movies and review it.

Here is a movie review: {movie_review}
Answer the following question based on the review above: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model
while True:
    print("\n\n-------------Movie Review Question Answering-------------\n\n")
    question = input("Enter your question about the movie review (or 'q' to quit): ")
    print("\n\n""-------------\n\n")
    if question == 'q':
        break
    reviews=retriever.invoke(question)
    result = chain.invoke({
        "movie_review": reviews,
        "question": question
    })

    print(result)
