import getpass
import os
import config
os.environ['USER_AGENT'] = 'myagent'

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://applied-llms.org/")


docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k':6})

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = hub.pull("rlm/rag-prompt")

user_query = "What is RAG?"
retrieved_docs = retriever.invoke("what is RAG?")
retrieved_docs_text = ""
for i in range(0,6) :
    doc_content = retrieved_docs[i].page_content
    retrieved_docs_text += f'{i+1}번 chunk: {doc_content}\n'

    query = {"question": {f"""
     너는 유저로부터 query 를 입력 받아서 최종적인 답변을 해주는 시스템이야.

     아래는 user query 와 retrieved chunk 에 대한 정보야.

     query : {user_query}\n
     retrieved chunks: {retrieved_docs_text}"""}}

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke({"query": query})
print(query)