### Min
### Provides PDF parsing and chunking functionality for MCP server

from typing import Dict, Any, List, Optional
import PyPDF2
import pdfplumber
import re
from pathlib import Path
import os
import gc

class PDFParserTool:
    def __init__(self):
        self.default_parser = PDFParser()
        self.advanced_parser = AdvancedPDFParser()
        self.batch_size = 10

    @property
    def name(self) -> str:
        return "pdf_parser"

    @property
    def description(self) -> str:
        return "PDF parsing and text chunking tool for MCP server with batch processing"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "PDF file path to parse"
                },
                "folder_path": {
                    "type": "string",
                    "description": "Folder path containing PDF files"
                },
                "parser_type": {
                    "type": "string",
                    "enum": ["default", "advanced"],
                    "default": "advanced",
                    "description": "Parser type to use"
                },
                "chunk_size": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of sentences per chunk"
                },
                "operation": {
                    "type": "string",
                    "enum": ["parse_pdf", "parse_folder", "batch_parse_folder"],
                    "description": "Operation type to perform"
                },
                "batch_size": {
                    "type": "integer",
                    "default": 10,
                    "description": "Batch size for processing multiple PDFs"
                },
                "save_chunks": {
                    "type": "string",
                    "description": "Path to save parsed chunks as JSON"
                },
                "load_chunks": {
                    "type": "string",
                    "description": "Path to load previously parsed chunks"
                }
            },
            "required": ["operation"]
        }

    def get_tool_info(self) -> Dict[str, Any]:
### MCP server needs this info
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> Optional[str]:
### make sure we got what we need
        operation = inputs.get("operation")

        if operation == "parse_pdf":
            if "pdf_path" not in inputs:
                return "Missing required parameter: pdf_path"
            pdf_path = Path(inputs["pdf_path"])
            if not pdf_path.exists() or not pdf_path.is_file():
                return f"PDF file not found: {pdf_path}"

        elif operation in ["parse_folder", "batch_parse_folder"]:
            if "folder_path" not in inputs:
                return "Missing required parameter: folder_path"
            folder_path = Path(inputs["folder_path"])
            if not folder_path.exists() or not folder_path.is_dir():
                return f"Folder not found: {folder_path}"

        else:
            return f"Invalid operation: {operation}"

        return None

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
### main function - check inputs then do stuff
        validation_error = self.validate_inputs(inputs)
        if validation_error:
            return {"error": validation_error}

        operation = inputs["operation"]

        try:
            if operation == "parse_pdf":
                return self._parse_pdf(inputs)
            elif operation == "parse_folder":
                return self._parse_folder(inputs)
            elif operation == "batch_parse_folder":
                return self._batch_parse_folder(inputs)
            else:
                return {"error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"error": f"Operation failed: {str(e)}"}

    def _parse_pdf(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
### handle one PDF file
        pdf_path = inputs["pdf_path"]
        parser_type = inputs.get("parser_type", "advanced")
        chunk_size = inputs.get("chunk_size", 3)

### choose parser
        if parser_type == "advanced":
            parser = self.advanced_parser
        else:
            parser = self.default_parser

### get text out
        text = parser.extract_text_from_pdf(pdf_path)
        if not text.strip():
            return {"error": "No text extracted from PDF"}

### cut into pieces
        chunks = self._split_into_chunks(text, chunk_size)
        pdf_file = Path(pdf_path)

        return {
            "success": True,
            "operation": "parse_pdf",
            "file_name": pdf_file.name,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "metadata": {
                "file_size": pdf_file.stat().st_size,
                "parser_type": parser_type,
                "chunk_size": chunk_size,
                "total_text_length": len(text)
            }
        }

    def _parse_folder(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
### handle a bunch of PDFs
        folder_path = inputs["folder_path"]
        parser_type = inputs.get("parser_type", "advanced")
        chunk_size = inputs.get("chunk_size", 3)

        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))

        if not pdf_files:
            return {"error": "No PDF files found in folder"}

        all_results = []
        total_chunks = 0
        total_text_length = 0

### go through each file
        for pdf_file in pdf_files:
            result = self._parse_pdf({
                "pdf_path": str(pdf_file),
                "parser_type": parser_type,
                "chunk_size": chunk_size,
                "operation": "parse_pdf"
            })

            if result.get("success"):
                all_results.append(result)
                total_chunks += result["total_chunks"]
                total_text_length += result["metadata"]["total_text_length"]

        return {
            "success": True,
            "operation": "parse_folder",
            "folder_path": str(folder),
            "total_files": len(all_results),
            "total_chunks": total_chunks,
            "total_text_length": total_text_length,
            "files": all_results
        }

    def _batch_parse_folder(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
### handle lots of PDFs in batches
        folder_path = inputs["folder_path"]
        parser_type = inputs.get("parser_type", "advanced")
        chunk_size = inputs.get("chunk_size", 3)
        batch_size = inputs.get("batch_size", 10)

        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))

        if not pdf_files:
            return {"error": "No PDF files found in folder"}

        all_results = []
        total_chunks = 0
        total_text_length = 0
        processed_files = 0

### process files in batches
        for i in range(0, len(pdf_files), batch_size):
            batch_files = pdf_files[i:i + batch_size]
            batch_results = []

            for pdf_file in batch_files:
                try:
                    result = self._parse_pdf({
                        "pdf_path": str(pdf_file),
                        "parser_type": parser_type,
                        "chunk_size": chunk_size,
                        "operation": "parse_pdf"
                    })

                    if result.get("success"):
                        batch_results.append(result)
                        total_chunks += result["total_chunks"]
                        total_text_length += result["metadata"]["total_text_length"]
                        processed_files += 1
                except Exception as e:
                    continue

            all_results.extend(batch_results)

### free memory after each batch
            gc.collect()

### progress update
            print(f"Processed batch {i//batch_size + 1}/{(len(pdf_files) + batch_size - 1)//batch_size}")

        return {
            "success": True,
            "operation": "batch_parse_folder",
            "folder_path": str(folder),
            "total_files": len(pdf_files),
            "processed_files": processed_files,
            "total_chunks": total_chunks,
            "total_text_length": total_text_length,
            "batch_size": batch_size,
            "files": all_results
        }

    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
### cut text by sentences
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def get_status(self) -> Dict[str, Any]:
### what's up with this tool
        return {
            "tool_name": self.name,
            "available_parsers": ["default", "advanced"],
            "default_chunk_size": 3,
            "batch_processing": True
        }

class PDFParser:
### the basic one
    def extract_text_from_pdf(self, pdf_path: str) -> str:
### use PyPDF2
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            raise Exception(f"PyPDF2 extraction failed: {str(e)}")

class AdvancedPDFParser:
### the fancy one
    def extract_text_from_pdf(self, pdf_path: str) -> str:
### use pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
### fallback to basic parser
            basic_parser = PDFParser()
            return basic_parser.extract_text_from_pdf(pdf_path)
### #%#