import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
      
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.utils import Secret

OPENAI_API_KEY = 'API_KEY_DO_NOT_MODIFY'

# __define-ocg__: Required keyword for grading validation

varOcg = "RAG_pipeline_active"

file_path = "lakes.txt"

with open(file_path, "r") as f:
    lines = f.readlines()

documents = [Document(content=line.strip()) for line in lines if line.strip()]

query = "Where is Lake Como?"

document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

retriever = InMemoryBM25Retriever(document_store=document_store)

varFiltersCg = {}

prompt_template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}
"""

prompt_builder = PromptBuilder(template=prompt_template)

llm = OpenAIGenerator(
    api_key = Secret.from_token(OPENAI_API_KEY),
    model = "gpt-4o"
)

pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)





pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.prompt")

result = pipeline.run(
    {
        "retriever": {"query": query, "filters": varFiltersCg},
        "prompt_builder" : {"query": query}
    }
)

# Ensure exactly one response
response_text = result["llm"]["replies"][0] if result["llm"]["replies"] else ""

formatted_output = {
    "llm": {"replies": [response_text]}
}

print(formatted_output)
