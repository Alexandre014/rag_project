import asyncio
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas import SingleTurnSample 
from ragas.metrics import ResponseRelevancy
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
async def evaluate_relevancy(scorer, sample):
    score = await scorer.single_turn_ascore(sample)
    print(f"The relevancy score is: {score}")

sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )

evaluator_llm = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")
scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings)
asyncio.run(evaluate_relevancy(scorer, sample))
