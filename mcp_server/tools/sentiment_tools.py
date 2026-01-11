from typing import Dict
from transformers import pipeline
from mcp_server.utils.text import clean_text, stars_to_binary


sentiment_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
)

def register(mcp) -> None:
    @mcp.tool()
    def analysis_of_sentiment(text: str) -> Dict:
        """
        Analyse le sentiment d'un texte (multilingue).

        Returns:
            {
            "stars": int (1..5),
            "binary_label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
            "score": float
            }
        """
        cleaned = clean_text(text)

        if not cleaned:
            return {"stars": 3, "binary_label": "NEUTRAL", "score": 0.0}

        result = sentiment_model(cleaned, truncation=True)[0]
        stars = int(result["label"][0])

        return {
            "stars": stars,
            "binary_label": stars_to_binary(result["label"]),
            "score": float(result["score"]),
        }
