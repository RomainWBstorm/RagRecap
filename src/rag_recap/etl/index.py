from rag_recap.config import settings, runner_settings
from rag_recap.etl.convert import convert_documents
from rag_recap.etl.chunk import get_chunker, iter_documents

from openai import OpenAI

from itertools import batched

def get_embedder(base_url, api_key):
    return OpenAI(
        base_url = base_url,
        api_key = api_key
    )

def embed(embedder, chunks, model_name):
    r = embedder.embeddings.create(model_name, input=chunks)
    return [v.embedding for v in r.data]

def iter_points(export_path, chunker, embedder, batch_size, embedder_model_name):
    chunks = iter_documents(export_path, chunker)
    for batch in batched(chunks, batch_size):
        embed(embedder, batch, embedder_model_name)

def index_all_documents():
    export_path = settings.export_path
    embedder_name = runner_settings['embedding']['model']
    max_tokens = runner_settings['chunking']['max_tokens']
    embedding_batch_size = settings.embedding_batch_size

    embedder_url = settings.embedder_url
    vllm_embedder_api_key = settings.vllm_embedder_api_key

    chunker = get_chunker(embedder_name, max_tokens)
    embedder = get_embedder(embedder_url, vllm_embedder_api_key)

    # create_collection()
    # init_collection()

    iter_points(
        export_path, chunker,
        embedder, embedding_batch_size,
        embedder_name
        )