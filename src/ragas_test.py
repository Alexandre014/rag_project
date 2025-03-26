import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset
from langchain_core.documents import Document
from ragas import evaluate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

content_list = [
    "Andrew Ng is the CEO of Landing AI and is known for his pioneering work in deep learning. He is also widely recognized for democratizing AI education through platforms like Coursera.",
    "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He is a strong advocate for creating safe and beneficial AI technologies.",
    "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems that can master complex games like AlphaGo.",
    "Sundar Pichai is the CEO of Google and Alphabet Inc., and he is praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences on a global scale.",
    "Arvind Krishna is the CEO of IBM and is recognized for transforming the company towards cloud computing and AI solutions. He focuses on providing cutting-edge technologies to address modern business challenges.",
]

langchain_documents = []

for content in content_list:
    langchain_documents.append(
        Document(
            page_content=content,
        )
    )
    
    


embeddings = OllamaEmbeddings(model="llama3")
vector_store = InMemoryVectorStore(embeddings)

_ = vector_store.add_documents(langchain_documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})




llm = ChatOllama(model="llama3")
#llm = Ollama(model="mistral", base_url=request.ollama_server_url)


template = """Answer the question based only on the following context:
{context}

Question: {query}
"""
prompt = ChatPromptTemplate.from_template(template)

qa_chain = prompt | llm | StrOutputParser()

def format_docs(relevant_docs):
    return "\n".join(doc.page_content for doc in relevant_docs)


query = "Who is the CEO of OpenAI?"

relevant_docs = retriever.invoke(query)
qa_chain.invoke({"context": format_docs(relevant_docs), "query": query})

sample_queries = [
    "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
    "Who is Sam Altman?",
    "Who is Demis Hassabis and how did he gained prominence?",
    "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
    "How did Arvind Krishna transformed IBM?",
]

expected_responses = [
    "Andrew Ng is the CEO of Landing AI and is widely recognized for democratizing AI education through platforms like Coursera.",
    "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He strongly advocates for creating safe and beneficial AI technologies.",
    "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems like AlphaGo that can master complex games.",
    "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences globally.",
    "Arvind Krishna is the CEO of IBM and has transformed the company towards cloud computing and AI solutions. He focuses on delivering cutting-edge technologies to address modern business challenges.",
]




dataset = []

for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = qa_chain.invoke({"context": format_docs(relevant_docs), "query": query})
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)



evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)

result