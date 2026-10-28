from rag_recap.config import settings, runner_settings

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

CHUNKERS = {
    'hybrid': HybridChunker,
    'hierarchical': HierarchicalChunker
}

CHUNKERS.get(runner_settings.get('type'))

def get_chunker():
    pass

def get_chunks(document, chunker):
    return [c for c in chunker.chunk(document)]