
import os

os.environ["OPENAI_API_KEY"] = "INSERT YOUR KEY"

import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output

from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore


from llama_index.llms import OpenAI
from IPython.display import Markdown, display


# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
llm = OpenAI(temperature=0, model="text-davinci-002")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

import os
import json
import openai
from langchain.embeddings import OpenAIEmbeddings
from llama_index.llms import AzureOpenAI
from llama_index import LangchainEmbedding
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    ServiceContext,
)
from llama_index import set_global_service_context

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

import logging
import sys

from IPython.display import Markdown, display

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


get_ipython().run_line_magic('pip', 'install nebula3-python ipython-ngql')

os.environ['NEBULA_USER'] = "root"
os.environ['NEBULA_PASSWORD'] = "nebula" # default password
os.environ['NEBULA_ADDRESS'] = "127.0.0.1:9669" # assumed we have NebulaGraph installed locally

space_name = "guardians"
edge_types, rel_prop_names = ["relationship"], ["relationship"] # default, could be omit if create from an empty kg
tags = ["entity"] # default, could be omit if create from an empty kg

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)


from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()

documents = loader.load_data(pages=['Guardians of the Galaxy Vol. 3'], auto_suggest=False)


kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
    max_triplets_per_chunk=10,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)


vector_index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)

# vector_index.storage_context.persist(persist_dir='./storage_vector')

from llama_index import load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir='./storage_graph', graph_store=graph_store)
kg_index = load_index_from_storage(
    storage_context=storage_context,
    service_context=service_context,
    max_triplets_per_chunk=10,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

storage_context_vector = StorageContext.from_defaults(persist_dir='./storage_vector')
vector_index = load_index_from_storage(
    service_context=service_context,
    storage_context=storage_context_vector
)


from llama_index.query_engine import KnowledgeGraphQueryEngine

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

nl2kg_query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)


kg_rag_query_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)


vector_rag_query_engine = vector_index.as_query_engine()


# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever

from typing import List


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

# create custom retriever
vector_retriever = VectorIndexRetriever(index=vector_index)
kg_retriever = KGTableRetriever(
    index=kg_index, retriever_mode="keyword", include_text=False
)
custom_retriever = CustomRetriever(vector_retriever, kg_retriever)

# create response synthesizer
response_synthesizer = get_response_synthesizer(
    service_context=service_context,
    response_mode="tree_summarize",
)


graph_vector_rag_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)


response_nl2kg = nl2kg_query_engine.query("Tell me about Peter Quill.")


display(Markdown(f"<b>{response_nl2kg}</b>"))

# Cypher:

print("Cypher Query:")

graph_query = nl2kg_query_engine.generate_query(
    "Tell me about Peter Quill?",
)
graph_query = graph_query.replace("WHERE", "\n  WHERE").replace("RETURN", "\nRETURN")

display(
    Markdown(
        f"""
```cypher
{graph_query}
```
"""
    )
)


response_graph_rag = kg_rag_query_engine.query("Tell me about Peter Quill.")

display(Markdown(f"<b>{response_graph_rag}</b>"))


response_vector_rag = vector_rag_query_engine.query("Tell me about Peter Quill.")

display(Markdown(f"<b>{response_vector_rag}</b>"))


display(
    Markdown(
        llm(f"""
Compare the two QA result on "Tell me about Peter Quill.", list the differences between them, to help evalute them. Output in markdown table.

Result from Graph: {response_graph_rag}
---
Result from Vector: {response_vector_rag}

"""
           )
    )
)


response_graph_vector_rag = graph_vector_rag_query_engine.query("Tell me about Peter Quill.")

display(Markdown(f"<b>{response_graph_vector_rag}</b>"))


display(
    Markdown(
        llm(f"""
Compare the QA results on "Tell me about Peter Quill.", list the knowledge facts between them, to help evalute them. Output in markdown table.

Result text2GraphQuery: {response_nl2kg}
---
Result Graph: {response_graph_rag}
---
Result Vector: {response_vector_rag}
---
Result Graph+Vector: {response_graph_vector_rag}
---

"""
           )
    )
)


display(
    Markdown(
        llm(f"""
Compare the two QA result on "Tell me about Peter Quill.", list the differences between them, to help evalute them. Output in markdown table.

Result from text-to-Graph: {response_nl2kg}
---
Result from Graph RAG: {response_graph_rag}

"""
           )
    )
)


graph_text2cypher = _


get_ipython().run_cell_magic('ngql', '', "MATCH path_0=(p:`entity`)-[*1..2]-()\n  WHERE p.`entity`.`name` == 'Peter Quill' \nRETURN path_0\n")


graph_rag = _


text2Cypher = """
<iframe
    src="https://www.siwei.io/demo-dumps/kg-llm/nebulagraph_draw_nl2cypher.html"
    width=450
    height=400>
</iframe>
"""
graphRAG = """
<iframe
    src="https://www.siwei.io/demo-dumps/kg-llm/nebulagraph_draw_rag.html"
    width=450
    height=400>
</iframe>
"""

table = f"""
<table>
    <tr>
        <th>Text2Cypher Traversed knowledge</th>
        <th>Graph Rag Traversed knowledge</th>
    </tr>
    <tr>
        <td>{text2Cypher}</td>
        <td>{graphRAG}</td>
    </tr>
</table>
"""

display(HTML(table))

