import os
import traceback

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from productassistant.utils import create_index_in_pinecone, read_file
import streamlit as st
from langchain.callbacks import get_openai_callback
from productassistant.logger import logging
from pathlib import Path
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

index_name = "textsearchindex3"
create_index_in_pinecone(index_name)

embed_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model = embed_model)
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)


###################################################################################
# Chat app code with chatting facility
###################################################################################
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
# chat_box = st.chat_input("")
if prompt := st.chat_input("Ask your question"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    docs = docsearch.similarity_search(prompt)
    llm = OpenAI()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever	= docsearch.as_retriever())
    llmresponse = qa.run(prompt)
    
    response = f"Echo: {llmresponse}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    
###################################################################################
# Browse pdf file and upload it in pdf directory
###################################################################################

if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = False
def show_upload(state:bool):
    st.session_state["uploader_visible"] = state
    
with st.chat_message("system"):
    cols= st.columns((3,1,1))
    cols[0].write("Do you want to upload a file?")
    cols[1].button("yes", use_container_width=True, on_click=show_upload, args=[True])
    cols[2].button("no", use_container_width=True, on_click=show_upload, args=[False])

if st.session_state["uploader_visible"]:
    with st.chat_message("system"):
        uploaded_file = st.file_uploader("Upload a pdf file")
        if uploaded_file:
            with st.spinner("Processing your file"):
                print(uploaded_file)
                save_folder = './pdfs'
                save_path = Path(save_folder, uploaded_file.name)
                print(save_path)
                with open(save_path, mode='wb') as w:
                    w.write(uploaded_file.getvalue())

                if save_path.exists():
                    st.success(f'File {uploaded_file.name} is successfully saved!')

text_chunks = read_file()
print(text_chunks)
docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)
                    
