from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

import tomllib

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    postgres_url: str
    qdrant_api_key: str

    # Servers
    ip_address:str = 'localhost'
    llm_url: str = f'http://{ip_address}:8000/v1'
    embedder_url: str = f'http://{ip_address}:8001/v1'
    reranker_url: str = f'http://{ip_address}:8002/rerank'

    # Paths
    corpus_path: Path = Path('/home/bstorm/rag_recap/data/hopital/hopital')
    export_path: Path = Path('/home/bstorm/rag_recap/data/export')
    docling_models_path: Path = Path('/home/bstorm/models/docling')

    # runner config
    runner_config_path: Path = Path('experiment/baseline.toml') 

class OcrSettings(BaseSettings):
    ocr_device: str = 'cpu'
    ocr_num_threads: int = 10
    tmp: int = 10

def get_runner_settings(config_path):
    with open(config_path, 'rb') as f:
        return tomllib.load(f)

settings = Settings()
ocr_settings = OcrSettings()
runner_settings = get_runner_settings(settings.runner_config_path)

if __name__ == '__main__':
    pass