"""chatlens: privacy-first WhatsApp chat parsing and analytics."""

from .config import AnalysisConfig
from .exceptions import ChatLensError, IngestionError, ParseError
from .ingestion import read_chat
from .parsing.parser import ParseReport, ParseResult, parse_chat
from .preprocessing.pipeline import preprocess
from .validation import QualityReport, validate

__version__ = "0.1.0"
__all__ = [
    "AnalysisConfig", "ChatLensError", "IngestionError", "ParseError", "ParseReport",
    "ParseResult", "QualityReport", "parse_chat", "preprocess", "read_chat", "validate",
]
