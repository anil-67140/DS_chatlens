class ChatLensError(Exception):
    """Base class for all chatlens errors."""


class IngestionError(ChatLensError):
    """The uploaded file was rejected (size, type, encoding, unsafe archive)."""


class ParseError(ChatLensError):
    """The text does not look like a WhatsApp export."""
