#!/usr/bin/env python3
"""
Interactive Docling Document Processor
Processes documents using various Docling pipelines with an interactive menu.
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json
import base64

# Set TESSDATA_PREFIX for Tesseract OCR if not already set
if 'TESSDATA_PREFIX' not in os.environ:
    tessdata_paths = [
        '/usr/share/tesseract/tessdata',
        '/usr/share/tessdata',
        '/usr/local/share/tessdata',
    ]
    for path in tessdata_paths:
        if Path(path).exists():
            os.environ['TESSDATA_PREFIX'] = path
            break

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import config, PIPELINES, PipelineConfig

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    RapidOcrOptions,
    EasyOcrOptions,
    TesseractOcrOptions,
    TesseractCliOcrOptions
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import PictureItem, TableItem


class DocumentProcessor:
    """Handles document processing with different pipelines."""

    def __init__(self, pipeline_key: str):
        self.pipeline_key = pipeline_key
        self.pipeline_config = config.get_pipeline_config(pipeline_key)
        if not self.pipeline_config:
            raise ValueError(f"Unknown pipeline: {pipeline_key}")

        self.converter: Optional[DocumentConverter] = None
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Set up the Docling pipeline based on configuration."""
        print(f"\n🔧 Setting up {self.pipeline_config.name}...")

        if self.pipeline_config.requires_models:
            print(f"   ⚠️  First run will download ~{self.pipeline_config.model_size_mb}MB of models")

        try:
            if self.pipeline_key == "simple":
                # Simple pipeline - no OCR, no models
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = False
                pipeline_options.do_table_structure = False
                pipeline_options.images_scale = 2.0
                pipeline_options.generate_picture_images = True

                self.converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )

            elif self.pipeline_key == "standard":
                # Standard pipeline with layout analysis and table detection
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = False
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                pipeline_options.images_scale = 2.0
                pipeline_options.generate_picture_images = True

                self.converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )

            elif self.pipeline_key == "rapidocr":
                # RapidOCR pipeline
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.ocr_options = RapidOcrOptions()
                pipeline_options.do_table_structure = True
                pipeline_options.images_scale = 2.0
                pipeline_options.generate_picture_images = True

                self.converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )

            elif self.pipeline_key == "easyocr":
                # EasyOCR pipeline
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.ocr_options = EasyOcrOptions()
                pipeline_options.do_table_structure = True
                pipeline_options.images_scale = 2.0
                pipeline_options.generate_picture_images = True

                self.converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )

            elif self.pipeline_key == "tesseract":
                # Tesseract pipeline - using CLI version for better compatibility
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.ocr_options = TesseractCliOcrOptions()
                pipeline_options.do_table_structure = True
                pipeline_options.images_scale = 2.0
                pipeline_options.generate_picture_images = True

                self.converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )

            elif self.pipeline_key == "vision_llm":
                # Vision LLM pipeline (will add image captioning post-processing)
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.ocr_options = RapidOcrOptions()
                pipeline_options.do_table_structure = True
                pipeline_options.images_scale = 2.0
                pipeline_options.generate_picture_images = True

                self.converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )

            elif self.pipeline_key == "formula_aware":
                # Formula-aware pipeline
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = False
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                pipeline_options.images_scale = 2.0
                pipeline_options.generate_picture_images = True

                self.converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )

            print("   ✓ Pipeline ready!")

        except Exception as e:
            print(f"   ✗ Error setting up pipeline: {e}")
            raise

    def process_document(self, doc_path: Path) -> Dict[str, Any]:
        """Process a document and return results."""
        print(f"\n📄 Processing: {doc_path.name}")
        start_time = time.time()

        try:
            # Convert document
            result = self.converter.convert(str(doc_path))
            doc = result.document

            # Extract statistics
            stats = self._extract_statistics(doc)

            # Export images from document and get captions if generated
            image_captions = self._export_images(doc, result, doc_path)

            # Save outputs
            self._save_outputs(doc, doc_path, stats, image_captions)

            elapsed = time.time() - start_time
            print(f"\n✓ Processing complete in {elapsed:.2f}s")

            return {"document": doc, "stats": stats, "elapsed": elapsed}

        except Exception as e:
            print(f"\n✗ Error processing document: {e}")
            if config.debug_mode:
                import traceback
                traceback.print_exc()
            raise

    def _export_images(self, doc, result, doc_path: Path):
        """Export figures and table images from the document."""
        if not config.extract_figures:
            return None

        base_name = doc_path.stem
        picture_counter = 0
        table_counter = 0
        image_metadata = []

        try:
            for element, _level in doc.iterate_items():
                # Export picture/figure items
                if isinstance(element, PictureItem):
                    picture_counter += 1
                    img_filename = config.assets_dir / f"{base_name}_fig{picture_counter}.png"
                    try:
                        with img_filename.open("wb") as fp:
                            element.get_image(doc).save(fp, "PNG")

                        # Store metadata for Vision LLM captioning
                        if self.pipeline_key == "vision_llm" and config.enable_image_captions:
                            image_metadata.append({
                                'path': img_filename,
                                'type': 'figure',
                                'number': picture_counter,
                                'element': element
                            })
                    except Exception as e:
                        if config.debug_mode:
                            print(f"   ⚠️  Error saving figure {picture_counter}: {e}")

                # Export table images
                if isinstance(element, TableItem):
                    table_counter += 1
                    img_filename = config.assets_dir / f"{base_name}_table{table_counter}.png"
                    try:
                        with img_filename.open("wb") as fp:
                            element.get_image(doc).save(fp, "PNG")
                    except Exception as e:
                        if config.debug_mode:
                            print(f"   ⚠️  Error saving table {table_counter}: {e}")

            if picture_counter > 0 or table_counter > 0:
                print(f"\n📸 Exported {picture_counter} figure(s) and {table_counter} table image(s) to assets/")

            # Generate Vision LLM captions if enabled
            captions = None
            if self.pipeline_key == "vision_llm" and config.enable_image_captions and image_metadata:
                captions = self._generate_image_captions(image_metadata, doc_path)

            return captions

        except Exception as e:
            print(f"   ⚠️  Error exporting images: {e}")
            if config.debug_mode:
                import traceback
                traceback.print_exc()
            return None

    def _generate_image_captions(self, image_metadata: list, doc_path: Path):
        """Generate AI descriptions for images using Vision LLM."""
        print(f"\n🤖 Generating AI descriptions for {len(image_metadata)} image(s)...")

        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.openai_api_key)

            captions = {}
            for img_data in image_metadata:
                img_path = img_data['path']
                img_type = img_data['type']
                img_num = img_data['number']

                try:
                    # Encode image to base64
                    with open(img_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                    # Call OpenAI Vision API
                    response = client.chat.completions.create(
                        model=config.vision_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Describe this image in detail. Focus on the technical content, diagrams, charts, or key information shown."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=config.vision_max_tokens
                    )

                    caption = response.choices[0].message.content
                    captions[f"{img_type}{img_num}"] = caption
                    print(f"   ✓ {img_type.capitalize()} {img_num} described")

                except Exception as e:
                    print(f"   ⚠️  Error describing {img_type} {img_num}: {e}")
                    if config.debug_mode:
                        import traceback
                        traceback.print_exc()

            # Save captions to file
            if captions:
                base_name = doc_path.stem
                captions_path = config.output_dir / f"{base_name}_captions.json"
                with open(captions_path, 'w', encoding='utf-8') as f:
                    json.dump(captions, f, indent=2, ensure_ascii=False)
                print(f"   💾 Captions saved to: {captions_path.name}")
                return captions

            return None

        except ImportError:
            print("   ⚠️  OpenAI library not found. Install with: pip install openai")
            return None
        except Exception as e:
            print(f"   ⚠️  Error generating captions: {e}")
            if config.debug_mode:
                import traceback
                traceback.print_exc()
            return None

    def _extract_statistics(self, doc) -> Dict[str, int]:
        """Extract statistics from the document."""
        stats = {
            "pages": 0,
            "text_blocks": 0,
            "tables": 0,
            "figures": 0,
            "formulas": 0,
        }

        try:
            # Count pages
            if hasattr(doc, 'pages'):
                stats["pages"] = len(doc.pages)

            # Try to get counts from document structure directly
            doc_dict = doc.export_to_dict()

            # Count texts
            if 'texts' in doc_dict:
                stats["text_blocks"] = len(doc_dict['texts'])

            # Count tables
            if 'tables' in doc_dict:
                stats["tables"] = len(doc_dict['tables'])

            # Count pictures/figures
            if 'pictures' in doc_dict:
                stats["figures"] = len(doc_dict['pictures'])

            # If we still have zero text blocks, try the old method as fallback
            if stats["text_blocks"] == 0:
                for item, _ in doc.iterate_items():
                    item_type = item.__class__.__name__

                    if "text" in item_type.lower() or "paragraph" in item_type.lower():
                        stats["text_blocks"] += 1
                    elif "table" in item_type.lower() and stats["tables"] == 0:
                        stats["tables"] += 1
                    elif "picture" in item_type.lower() or "figure" in item_type.lower():
                        if stats["figures"] == 0:
                            stats["figures"] += 1
                    elif "formula" in item_type.lower() or "equation" in item_type.lower():
                        stats["formulas"] += 1

        except Exception as e:
            print(f"   ⚠️  Error extracting statistics: {e}")
            if config.debug_mode:
                import traceback
                traceback.print_exc()

        return stats

    def _save_outputs(self, doc, doc_path: Path, stats: Dict[str, int], image_captions: Optional[Dict] = None):
        """Save document outputs in various formats."""
        base_name = doc_path.stem

        # Save JSON (DoclingDocument serialization)
        json_path = config.output_dir / f"{base_name}_docling.json"
        print(f"\n💾 Saving outputs...")
        print(f"   → JSON: {json_path.name}")

        try:
            # Export to dict and save as JSON
            doc_dict = doc.export_to_dict()
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"   ⚠️  Error saving JSON: {e}")

        # Save Markdown preview
        md_path = config.output_dir / f"{base_name}_preview.md"
        print(f"   → Markdown: {md_path.name}")

        try:
            self._generate_markdown_preview(doc, md_path, doc_path.name, stats, image_captions)
        except Exception as e:
            print(f"   ⚠️  Error saving Markdown: {e}")

    def _generate_markdown_preview(self, doc, output_path: Path, doc_name: str, stats: Dict[str, int], image_captions: Optional[Dict] = None):
        """Generate a comprehensive Markdown preview of the document."""
        base_name = Path(doc_name).stem

        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Document: {doc_name}\n\n")
            f.write(f"**Pipeline:** {self.pipeline_config.name}\n\n")
            f.write(f"**Processed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Statistics
            f.write("## Statistics\n\n")
            f.write(f"- Pages: {stats['pages']}\n")
            f.write(f"- Text blocks: {stats['text_blocks']}\n")
            f.write(f"- Tables: {stats['tables']}\n")
            f.write(f"- Figures: {stats['figures']}\n")
            f.write(f"- Formulas: {stats['formulas']}\n\n")

            # AI Image Descriptions (if available)
            if image_captions and stats['figures'] > 0:
                f.write("## AI Image Descriptions\n\n")
                f.write("*Generated by Vision LLM (GPT-4o)*\n\n")

                for i in range(1, stats['figures'] + 1):
                    caption_key = f"figure{i}"
                    if caption_key in image_captions:
                        img_path = f"../assets/{base_name}_fig{i}.png"
                        f.write(f"### Figure {i}\n\n")
                        f.write(f"**Image:** [{base_name}_fig{i}.png]({img_path})\n\n")
                        f.write(f"**Description:** {image_captions[caption_key]}\n\n")
                        f.write("---\n\n")

            f.write("---\n\n")

            # Content
            f.write("## Content\n\n")

            # Export as markdown using Docling's built-in export
            try:
                markdown_content = doc.export_to_markdown()

                # Check if markdown export is meaningful (more than just comments/whitespace)
                has_content = len(markdown_content.strip()) > 20 and not markdown_content.strip().startswith('<!--')

                if has_content:
                    f.write(markdown_content)
                else:
                    # Markdown export was empty or minimal, use manual extraction
                    doc_dict = doc.export_to_dict()

                    # Extract and write text blocks
                    if 'texts' in doc_dict and doc_dict['texts']:
                        for i, text_item in enumerate(doc_dict['texts'], 1):
                            text_content = text_item.get('text', '').strip()
                            if text_content:
                                f.write(f"{text_content}\n\n")
                    else:
                        f.write("*No text content found in document*\n\n")

                    # Note if there are figures
                    if stats['figures'] > 0:
                        f.write(f"\n*[Document contains {stats['figures']} figure(s) - see assets/ folder]*\n\n")

            except Exception as e:
                f.write(f"*Error generating markdown content: {e}*\n\n")

                # Final fallback: try to extract from JSON
                try:
                    doc_dict = doc.export_to_dict()
                    if 'texts' in doc_dict and doc_dict['texts']:
                        f.write("### Extracted Text\n\n")
                        for text_item in doc_dict['texts']:
                            text_content = text_item.get('text', '').strip()
                            if text_content:
                                f.write(f"{text_content}\n\n")
                except Exception as e2:
                    f.write(f"*Error extracting text: {e2}*\n")


def display_banner():
    """Display application banner."""
    print("\n" + "=" * 60)
    print("  DOCLING DOCUMENT PROCESSOR")
    print("  Interactive Document Ingestion Tool")
    print("=" * 60)


def select_document() -> Optional[Path]:
    """Display document selection menu."""
    documents = config.get_available_documents()

    if not documents:
        print("\n⚠️  No documents found in 'documents/' directory")
        print(f"   Please add PDF, DOCX, PPTX, or HTML files to: {config.documents_dir}")
        return None

    print("\n📂 Available documents:")
    for idx, doc in enumerate(documents, 1):
        size_mb = doc.stat().st_size / (1024 * 1024)
        print(f"   {idx}. {doc.name} ({size_mb:.2f} MB)")
    print("   0. Exit")

    while True:
        try:
            choice = input("\nSelect document number: ").strip()
            if choice == "0":
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(documents):
                return documents[idx]
            else:
                print("   ⚠️  Invalid selection. Try again.")
        except ValueError:
            print("   ⚠️  Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return None


def select_pipeline() -> Optional[str]:
    """Display pipeline selection menu."""
    available_pipelines = config.get_available_pipelines()

    if not available_pipelines:
        print("\n⚠️  No pipelines available")
        return None

    print("\n🔧 Available Pipelines:")
    pipeline_list = list(available_pipelines.items())

    for idx, (key, pipeline) in enumerate(pipeline_list, 1):
        marker = "⚡" if not pipeline.requires_models else "📦"
        api_marker = " 🔑" if pipeline.requires_api_key else ""
        print(f"   {idx}. {marker} {pipeline.name}{api_marker}")
        print(f"      {pipeline.description}")

    print("   0. Back")

    while True:
        try:
            choice = input("\nSelect pipeline number: ").strip()
            if choice == "0":
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(pipeline_list):
                return pipeline_list[idx][0]
            else:
                print("   ⚠️  Invalid selection. Try again.")
        except ValueError:
            print("   ⚠️  Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nReturning to main menu...")
            return None


def main():
    """Main application loop."""
    display_banner()

    while True:
        # Select document
        doc_path = select_document()
        if not doc_path:
            print("\nGoodbye! 👋")
            sys.exit(0)

        # Select pipeline
        pipeline_key = select_pipeline()
        if not pipeline_key:
            continue  # Go back to document selection

        # Process document
        try:
            processor = DocumentProcessor(pipeline_key)
            result = processor.process_document(doc_path)

            print("\n" + "=" * 60)
            print("  PROCESSING SUMMARY")
            print("=" * 60)
            print(f"Document: {doc_path.name}")
            print(f"Pipeline: {processor.pipeline_config.name}")
            print(f"Time: {result['elapsed']:.2f}s")
            print(f"\nStatistics:")
            for key, value in result['stats'].items():
                print(f"  - {key.capitalize()}: {value}")
            print("\n✓ Outputs saved to: out/")
            print("=" * 60)

        except Exception as e:
            print(f"\n✗ Processing failed: {e}")
            if config.debug_mode:
                import traceback
                traceback.print_exc()

        # Ask if user wants to process another document
        print("\n")
        choice = input("Process another document? (y/n): ").strip().lower()
        if choice != 'y':
            print("\nGoodbye! 👋")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye! 👋")
        sys.exit(0)
