import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os


chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# Load vectorstore
embedding = OpenAIEmbeddings()
vectordb = FAISS.load_local("union_embeddings", embedding)
docsearch = vectordb.as_retriever()

system_template = """I am a stakeholder interested in a teacher's union contract. 
Use the following text from the contract to answer the question at the end. If you don't know the answer, 
just say that and provide a summary of the relevant information from the information below that I will 
need to answer my question myself.

{context}

---
Also share the relevant quotes from the contract, and the section and page number.

Format your answer like this: 

<answer>
source:
<text of section you used to answer the question>
<page #(s)>"""

human_template = "My question is: {question}"

st.set_page_config(
    page_title="chatUTR",
    page_icon="🤖",
)

# hide streamlit branding
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🗃️ chatUTR 🤖")
st.markdown("""Question/answer for UTR contract questions""")

with st.form(key="query_form"):
    user_question = st.text_input("Enter your contract question:","Am I eligible for parental leave?")
    submit_button = st.form_submit_button("Submit")
    if submit_button:
        docs = docsearch.get_relevant_documents(user_question)
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])
        result = chat(chat_prompt.format_prompt(context=docs, question=user_question).to_messages())
        st.markdown(result.content)