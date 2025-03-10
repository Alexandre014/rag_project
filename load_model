import sys
import asyncio
import tkinter as tk
from tkinter import filedialog
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


checkpoint = "mistralai/Mistral-7B-v0.1"

token = "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"


tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = token, low_cpu_mem_usage=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained("model")


model = AutoModelForCausalLM.from_pretrained(checkpoint, token = token, low_cpu_mem_usage=True) #model for generation
model.save_pretrained("model", max_shard_size="2GB")