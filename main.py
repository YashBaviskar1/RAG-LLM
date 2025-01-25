import streamlit as st 
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint

import os 
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore() :
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template) :
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "input"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN) :
    llm = HuggingFaceEndpoint(repo_id = huggingface_repo_id, temperature=0.5, model_kwargs={"token" : HF_TOKEN, "max_length" : "512"})
    return llm 




def main() :
    st.title("Mental Health Chatbot!")

    if 'messages' not in st.session_state :
        st.session_state.messages = []

    for message in st.session_state.messages :
        st.chat_message(message['role']).markdown(message['content'])
    prompt = st.chat_input("Ask your query here : ")
    if prompt :
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role' : "user", "content" : prompt})
        CUSTOM_PROMPT_TEMPLATE = """
        Use the information in context to answer to user's question 
        Strictly stay within the context and do not give answer of things you do not know, 
        do not make up answer and you can say i do not know if you do not know the answer 
        Do not provide anything out of the given context 

        Context : {context}

        Question : {input}

        Be extensive and accurate, and do not say things like "In the context provided" or tlak about context, just give answer to the question
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
        db = get_vectorstore()
        retriever = db.as_retriever(search_kwargs={'k': 3})
        
        prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
        qa_document_chain = create_stuff_documents_chain(llm, prompt_template)
        qa_chain = create_retrieval_chain(retriever, qa_document_chain)
        response = qa_chain.invoke({"input": prompt})
        st.chat_message('ai').markdown(response['answer'])
        st.session_state.messages.append({'role' : "ai", "content" : response['answer']})

if __name__ == '__main__' :
    main()