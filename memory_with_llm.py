from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
import os 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id) :
    llm = HuggingFaceEndpoint(repo_id = huggingface_repo_id, temperature=0.5, model_kwargs={"token" : HF_TOKEN, "max_length" : "512"})
    return llm 


CUSTOM_PROMPT_TEMPLATE = """
Use the information in context to answer to user's question 
Strictly stay within the context and do not give answer of things you do not know, 
do not make up answer and you can say i do not know if you do not know the answer 
Do not provide anything out of the given context 

Context : {context}

Question : {input}

Be extensive and accurate
"""

def set_custom_prompt(custom_prompt_template) :
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "input"])
    return prompt

DB_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)


retriever = db.as_retriever(search_kwargs={'k': 3})
llm = load_llm(HUGGINGFACE_REPO_ID)
prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
qa_document_chain = create_stuff_documents_chain(llm, prompt_template)


qa_chain = create_retrieval_chain(retriever, qa_document_chain)

# user_query = input("Write query here : ")
# response = qa_chain.invoke({"input": user_query})
# print(response["answer"])
# print(response)


def get_response(user_query) : 
    response = qa_chain.invoke({"input": user_query})
    #print(response["answer"])
    return response['answer']

