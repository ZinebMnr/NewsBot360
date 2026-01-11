from typing import List
from PyPDF2 import PdfReader
from mcp_server.utils.text import clean_text

def register(mcp) -> None:
    @mcp.tool()
    def pdf_to_page_list(pdf_path: str) -> List[str]:
        """
        Reads a PDF file and returns a list of strings, one per page.
        """
        reader = PdfReader(pdf_path)
        return [clean_text(p.extract_text()) for p in reader.pages]