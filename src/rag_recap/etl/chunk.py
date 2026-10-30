from rag_recap.config import settings, runner_settings

from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

from transformers import Autotokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

CHUNKERS = {
    'hybrid': HybridChunker,
    'hierarchical': HierarchicalChunker
}

def get_chunker(chunker_type, embedder_name, max_tokens):
    tokenizer = HuggingFaceTokenizer(
        tokenizer = Autotokenizer.from_pretrained(embedder_name),
        max_tokens = max_tokens
    )

    chunker = CHUNKERS[chunker_type](tokenizer = tokenizer)
    return chunker

def chunk_document(path, chunker):
    doc = DoclingDocument.load_from_json(path)
    for i, chunk in enumerate(chunker.chunk(dl_doc = doc)):
        yield {
            'document': path.stem,
            'chunk_index': i,
            'headings': ' | '.join(chunk.meta.headings or []),
            'text': chunker.contextualize(chunk),
            'file_extension': path.suffixes[-2].lstrip('.')
        }

def iter_documents(export_path, chunker):
    for document_path in export_path.rglob('*.json'):
        yield from chunk_document(document_path, chunker)