from supabase import create_client
import os
import pandas as pd


supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)


def recup_traitement_date_bdd(date_debut, date_fin, theme) -> pd.DataFrame:
    response = (
        supabase
        .table("articles")
        .select("title")
        .gte("published_at", date_debut)
        .lte("published_at", date_fin)
        .ilike("title", f"%{theme}%")
        .execute()
)
    return pd.DataFrame(response.data)

def register(mcp) -> None:
    @mcp.tool()
    def news_theme_date(date_debut,date_fin,theme):
        """
        Prend en entré une date de début et de fin ( format '2026-01-31'),
        et un thème (pas plus de 2 mots) et renvoi les titres des articles sur ce thème durant cette periode.
        """
        date_debut = f"{date_debut}T00:00:00"
        date_fin = f"{date_fin}T23:59:59"
        df = recup_traitement_date_bdd(date_debut=date_debut,date_fin=date_fin,theme=theme)
        titles = df["title"].tolist()
        return titles