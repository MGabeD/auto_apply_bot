import magic
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from auto_apply_bot.utils.logger import get_logger


logger = get_logger(__name__)
mime_detector = magic.Magic(mime=True)


def get_mime_type(file) -> str:
    pos = file.tell()
    sample = file.read(2048)
    if not hasattr(file, 'read') or not hasattr(file, 'seek'):
        raise ValueError("Invalid file object")
    file.seek(pos)
    return mime_detector.from_buffer(sample)


def validate_file(file, allowed_extensions: Dict[str, List[str]], max_size: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    ext = Path(file.name).suffix.lower()
    if ext not in allowed_extensions:
        logger.warning(f"Rejected file {file.name}: unsupported extension {ext}")
        return False, f"Invalid file extension: {ext}"

    expected_mime_type = allowed_extensions[ext]
    if not isinstance(expected_mime_type, list):
        logger.warning(f"Expected mime type for extension {ext} is not a list, converting to list")
        expected_mime_type = [expected_mime_type]

    mime_type = get_mime_type(file)
    if mime_type not in expected_mime_type:
        if not any(mime_type.startswith(allowed_mime) for allowed_mime in expected_mime_type):
            logger.warning(f"Invalid mime type: {mime_type}, expected: {expected_mime_type}")
            return False, f"Expected MIME type: {expected_mime_type}, got: {mime_type}"
        else:
            logger.info(f"FUZZY MATCH: Mime type {mime_type} is not an exact match for expected mime types: {expected_mime_type}")
    file_size = getattr(file, 'size', 0)
    if max_size and file_size > max_size:
        logger.warning(f"File size exceeds maximum allowed size: {max_size} bytes")
        return False, f"File size exceeds maximum allowed size: {max_size} bytes"

    return True, None
