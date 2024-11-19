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

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
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

user_query = "What is agent memory?"
retrieved_docs = retriever.invoke("what is agent memory?")
retrieved_docs_text = ""
for i in range(0,6) :
    doc_content = retrieved_docs[i].page_content
    retrieved_docs_text += f'{i+1}번 chunk: {doc_content}\n'

    query = {"question": {f"""
     너는 유저로부터 query 를 입력 받아서 최종적인 답변을 해주는 시스템이야.

     그런데, retrieval 된 결과를 본 후에, relevance 가 있는 retrieved chunk 만을 참고해서 최종적인 답변을 해줄거야.
     아래 step 대로 수행한 후에 최종적인 답변을 출력해.

     Step1) retrieved chunk 를 보고 관련이 있으면 각각의 청크 번호별로 relevance: yes 관련이 없으면 relevance: no 의 형식으로 json 방식으로 출력해.\n
     Step2) 만약 모든 chunk 가 relevance 가 no 인 경우에, relevant chunk 를 찾을 수 없었습니다. 라는 메시지와 함께 프로그램을 종료해.
     Step3) 만약 하나 이상의 chunk 가 relevance 가 yes 인 경우에, 그 chunk들만 참고해서 답변을 생성해.
     Step4) 최종 답변에 hallucination 이 있었는지를 평가하고 만약, 있었다면 hallucination : yes 라고 출력해. 없다면 hallucination : no 라고 출력해.
     Step5) 만약 hallucination : no 인 경우에, Step3로 돌아가서 답변을 다시 생성해. 이 과정은 딱 1번만 실행해 무한루프가 돌지 않도록.
     Step6) 지금까지의 정보를 종합해서 최종적인 답변을 생성해

     답변은 각 스텝별로 상세하게 출력해

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
