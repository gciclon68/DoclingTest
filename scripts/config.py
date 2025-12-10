"""
Configuration module for Docling document processing pipelines.
Loads settings from .Docling_env and .Key_env files.
"""

import os
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment files
load_dotenv(PROJECT_ROOT / ".Docling_env")
load_dotenv(PROJECT_ROOT / ".Key_env")


@dataclass
class PipelineConfig:
    """Configuration for a Docling processing pipeline."""
    name: str
    key: str
    description: str
    requires_api_key: bool = False
    requires_models: bool = False
    supports_ocr: bool = False
    supports_vision: bool = False
    model_size_mb: Optional[int] = None


# Define all available pipelines with numbers for better organization
PIPELINES = {
    "simple": PipelineConfig(
        name="1-Simple PDF",
        key="SIMPLE_PDF",
        description="Fast PDF processing, no ML models, no OCR, basic text extraction",
        requires_api_key=False,
        requires_models=False,
        supports_ocr=False,
        supports_vision=False,
        model_size_mb=0
    ),
    "standard": PipelineConfig(
        name="2-Standard PDF",
        key="STANDARD_PDF",
        description="Full-featured with layout analysis and table detection (downloads ML models)",
        requires_api_key=False,
        requires_models=True,
        supports_ocr=False,
        supports_vision=False,
        model_size_mb=500
    ),
    "rapidocr": PipelineConfig(
        name="3-RapidOCR",
        key="RAPIDOCR_PIPELINE",
        description="OCR-enabled with RapidOCR (fast, local, good for scanned docs)",
        requires_api_key=False,
        requires_models=True,
        supports_ocr=True,
        supports_vision=False,
        model_size_mb=50
    ),
    "easyocr": PipelineConfig(
        name="4-EasyOCR",
        key="EASYOCR_PIPELINE",
        description="OCR with EasyOCR (slower but better accuracy, 80+ languages)",
        requires_api_key=False,
        requires_models=True,
        supports_ocr=True,
        supports_vision=False,
        model_size_mb=200
    ),
    "tesseract": PipelineConfig(
        name="5-Tesseract",
        key="TESSERACT_PIPELINE",
        description="OCR with Tesseract (traditional, requires system install - may not work)",
        requires_api_key=False,
        requires_models=False,
        supports_ocr=True,
        supports_vision=False,
        model_size_mb=0
    ),
    "vision_llm": PipelineConfig(
        name="6-VisionLLM",
        key="VISION_LLM_PIPELINE",
        description="Advanced OCR + AI image descriptions (requires API key, costs apply)",
        requires_api_key=True,
        requires_models=True,
        supports_ocr=True,
        supports_vision=True,
        model_size_mb=500
    ),
    "formula_aware": PipelineConfig(
        name="7-FormulaAware",
        key="FORMULA_AWARE_PIPELINE",
        description="Specialized for math documents, preserves LaTeX formulas and symbols",
        requires_api_key=False,
        requires_models=True,
        supports_ocr=False,
        supports_vision=False,
        model_size_mb=500
    ),
}


class Config:
    """Main configuration class for the application."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.documents_dir = PROJECT_ROOT / "documents"
        self.output_dir = PROJECT_ROOT / "out"
        self.assets_dir = PROJECT_ROOT / "assets"

        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)

        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Vision model settings
        self.vision_model = os.getenv("VISION_MODEL", "gpt-4o")
        self.vision_max_tokens = int(os.getenv("VISION_MAX_TOKENS", "200"))

        # Feature flags
        self.enable_image_captions = os.getenv("ENABLE_IMAGE_CAPTIONS", "true").lower() == "true"
        self.extract_figures = os.getenv("EXTRACT_FIGURES", "true").lower() == "true"
        self.preserve_latex = os.getenv("PRESERVE_LATEX", "true").lower() == "true"
        self.generate_toc = os.getenv("GENERATE_TOC", "true").lower() == "true"
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

    def has_api_key(self, provider: str = "openai") -> bool:
        """Check if API key is available for the given provider."""
        if provider.lower() == "openai":
            return bool(self.openai_api_key and self.openai_api_key != "your_openai_api_key_here")
        elif provider.lower() == "anthropic":
            return bool(self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_api_key_here")
        return False

    def get_available_documents(self, folder_path: Optional[Path] = None) -> list[Path]:
        """Get list of documents in the specified folder or documents directory.

        Args:
            folder_path: Optional Path to a subfolder within documents/.
                        If None, searches the root documents/ directory.

        Returns:
            Sorted list of document Paths in the specified folder (non-recursive).
        """
        # Default to documents root if no folder specified
        search_dir = folder_path if folder_path else self.documents_dir

        if not search_dir.exists():
            return []

        # Validate that search_dir is within documents directory (prevent path traversal)
        try:
            search_dir.resolve().relative_to(self.documents_dir.resolve())
        except ValueError:
            # Path is outside documents directory
            return []

        # Supported file extensions
        extensions = {'.pdf', '.docx', '.pptx', '.html', '.htm', '.xml', '.md'}

        documents = []
        for ext in extensions:
            # Only search immediate directory, not recursive
            documents.extend(search_dir.glob(f"*{ext}"))

        return sorted(documents)

    def get_subfolders(self, folder_path: Optional[Path] = None) -> list[Path]:
        """Get list of immediate subdirectories within a given folder.

        Args:
            folder_path: Optional Path to search within. If None, searches documents/ root.

        Returns:
            Sorted list of subdirectory Paths (non-hidden folders only).
        """
        search_dir = folder_path if folder_path else self.documents_dir

        if not search_dir.exists():
            return []

        # Validate path is within documents directory
        try:
            search_dir.resolve().relative_to(self.documents_dir.resolve())
        except ValueError:
            return []

        # Get immediate subdirectories, filter out hidden folders
        subfolders = [
            d for d in search_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]

        return sorted(subfolders)

    def get_relative_output_dir(self, doc_path: Path) -> Path:
        """Calculate output directory for a document, mirroring folder structure.

        Args:
            doc_path: Path to the document file.

        Returns:
            Path to the output directory for this document.

        Raises:
            ValueError: If path is too deep or invalid.
        """
        # Calculate relative path from documents/ to document's parent folder
        try:
            relative = doc_path.resolve().relative_to(self.documents_dir.resolve()).parent
        except ValueError:
            # Document is outside documents directory, use default
            return self.output_dir

        # Validate path depth (max 10 levels)
        if len(relative.parts) > 10:
            raise ValueError(f"Path too deep (max 10 levels): {relative}")

        # If document is in root (relative is "."), use output_dir directly
        if str(relative) == ".":
            return self.output_dir

        # Create mirrored folder structure
        output_dir = self.output_dir / relative
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def get_relative_assets_dir(self, doc_path: Path) -> Path:
        """Calculate assets directory for a document, mirroring folder structure.

        Args:
            doc_path: Path to the document file.

        Returns:
            Path to the assets directory for this document.

        Raises:
            ValueError: If path is too deep or invalid.
        """
        # Calculate relative path from documents/ to document's parent folder
        try:
            relative = doc_path.resolve().relative_to(self.documents_dir.resolve()).parent
        except ValueError:
            # Document is outside documents directory, use default
            return self.assets_dir

        # Validate path depth (max 10 levels)
        if len(relative.parts) > 10:
            raise ValueError(f"Path too deep (max 10 levels): {relative}")

        # If document is in root (relative is "."), use assets_dir directly
        if str(relative) == ".":
            return self.assets_dir

        # Create mirrored folder structure
        assets_dir = self.assets_dir / relative
        assets_dir.mkdir(parents=True, exist_ok=True)

        return assets_dir

    def validate_batch_preconditions(self, docs: list[Path]) -> list[str]:
        """Validate documents before batch processing and return warnings.

        Args:
            docs: List of document Paths to validate.

        Returns:
            List of warning/error messages (empty if all good).
        """
        warnings = []

        for doc in docs:
            # Check if document exists
            if not doc.exists():
                warnings.append(f"File not found: {doc.name}")
                continue

            # Warn about very large files (>500MB)
            file_size = doc.stat().st_size
            if file_size > 500 * 1024 * 1024:  # 500MB
                size_mb = file_size / (1024 * 1024)
                warnings.append(f"Very large file (may be slow): {doc.name} ({size_mb:.1f} MB)")

        return warnings

    def get_pipeline_config(self, pipeline_key: str) -> Optional[PipelineConfig]:
        """Get configuration for a specific pipeline."""
        return PIPELINES.get(pipeline_key)

    def get_available_pipelines(self) -> Dict[str, PipelineConfig]:
        """Get all available pipelines, filtering out those requiring unavailable API keys."""
        available = {}
        for key, pipeline in PIPELINES.items():
            # If pipeline requires API key, check if we have it
            if pipeline.requires_api_key:
                if not (self.has_api_key("openai") or self.has_api_key("anthropic")):
                    continue  # Skip this pipeline
            available[key] = pipeline
        return available


# Global config instance
config = Config()
