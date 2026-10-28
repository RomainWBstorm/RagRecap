from rag_recap.config import settings, ocr_settings

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions

import hashlib

ALLOWED_FILE_EXTENSION = [InputFormat.PDF, InputFormat.DOC, InputFormat.DOCX]
ALLOWED_SUFFIXES = {'.pdf', '.docx', '.doc'}
OCR = False

def get_file_list(corpus_path):
    # corpus_path.rglob()
    # yield from [file for file in corpus_path.iterdir() if file.suffix.lower() in ALLOWED_SUFFIXES]
    return [file for file in corpus_path.iterdir() if file.suffix.lower() in ALLOWED_SUFFIXES]    

def get_converter():
    if not OCR:
        return DocumentConverter()

    ocr_options = PdfPipelineOptions(
        artifacts_path = settings.docling_models_path,
        accelerator_options = AcceleratorOptions(
            device = ocr_settings.ocr_device,
            num_threads = ocr_settings.ocr_num_threads
        )
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(ocr_options)}
    )

    return converter

def get_hash(file_path):
    return hashlib.sha256(file_path.read_bytes()).hexdigest()
    
def convert_documents():
    converter = get_converter()
    settings.export_path.mkdir(exist_ok=True)

    for file_path in get_file_list(settings.corpus_path):
        doc = converter.convert(file_path).document
        file_hash = get_hash(file_path)
        doc.save_as_json(settings.export_path / (file_hash + '.json'))
        doc.save_as_markdown(settings.export_path / (file_hash + '.md'))

if __name__ == '__main__':
    convert_documents()
    