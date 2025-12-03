import base64
from fastapi import UploadFile

def image_to_base64(file: UploadFile) -> str:
    content = file.file.read()
    return base64.b64encode(content).decode("utf-8")
