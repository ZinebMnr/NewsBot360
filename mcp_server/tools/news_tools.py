from typing import List, Dict, Optional
import requests
from mcp_server.config import NEWSAPI_KEY, NEWSAPI_URL

def register(mcp) -> None:
    @mcp.tool()
    def newsbytheme(topic: str) -> List[Dict[str, Optional[str]]]:
        """
        Tool MCP : récupère 20 articles NewsAPI liés à un thème.
        """
        if not NEWSAPI_KEY:
            return [{
                "error": "MISSING_API_KEY",
                "message": "NEWSAPI_KEY non trouvée"
            }]

        params = {
            "q": topic,
            "language": "fr",
            "pageSize": 20,
            "sortBy" : "publishedAt",
            "apiKey": NEWSAPI_KEY,
        }

        try:
            r = requests.get(NEWSAPI_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return [{"error": "HTTP_ERROR", "message": str(e)}]

        if data.get("status") != "ok":
            return [{"error": "NEWSAPI_ERROR", "message": str(data)}]

        return [
            {
                "title": a.get("title"),
                "source": (a.get("source") or {}).get("name"),
                "published_at": a.get("publishedAt"),
                "url": a.get("url"),
                "description": a.get("description"),
            }
            for a in data.get("articles", [])
        ]
