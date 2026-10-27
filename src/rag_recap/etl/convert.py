from rag_recap.config import settings, ocr_settings

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions

ALLOWED_FILE_EXTENSION = [InputFormat.PDF, InputFormat.DOC, InputFormat.DOCX]
OCR = False

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

def convert_documents():
    converter = get_converter()

    converter.convert(settings.corpus_path)

    settings.export_path.mkdir(exist_ok=True)
    