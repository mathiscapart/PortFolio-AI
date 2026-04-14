class StructuredData:
    def __init__(self):
        self.data = None

    def pdf_to_text(self, pdf_path: str) -> str:
        return "Extracted text from PDF"

    def file_to_structured(self, file_path: str) -> dict:
        with open(file_path, 'r') as f:
            self.data = f.read()
        return self.data
    
    def text_to_structured(self, text: str) -> dict:
        self.data = text
        return self.data
    