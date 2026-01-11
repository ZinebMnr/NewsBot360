from transformers import pipeline
import requests
import trafilatura

HEADERS = {"User-Agent": "Mozilla/5.0"}

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

def register(mcp) -> None:
    @mcp.tool()
    def summarize_link(link: str):
        """
        Prend un lien d'un article de la bbc, de franceinfo ou de The Guardian
        """
        session = requests.Session()
        r = session.get(link, headers=HEADERS, timeout=10, allow_redirects=True)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        content = trafilatura.extract(r.text)
        content = content.strip()[:4000]
        result = summarizer(
            content,
            max_length=90,
            do_sample=False,
            num_beams=6,
            length_penalty=1.1
        )
        return(result[0]["summary_text"])