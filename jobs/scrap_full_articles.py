import os
import time
import requests
import trafilatura
from supabase import create_client

FEED_ID = 3
HEADERS = {"User-Agent": "Mozilla/5.0"}

def scrape_one(session: requests.Session, link: str, timeout=10, min_chars=400):
    r = session.get(link, headers=HEADERS, timeout=timeout, allow_redirects=True)
    if r.status_code != 200:
        return None, f"http_{r.status_code}"

    content = trafilatura.extract(r.text)
    if not content or len(content) < min_chars:
        return None, "text_too_short_or_none"

    return content, None


def main():
    # --- Supabase client ---
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ SUPABASE_URL ou SUPABASE_KEY manquant")
        return 1

    supabase = create_client(supabase_url, supabase_key)

    # 1) Articles du feed
    articles = (
        supabase
        .table("articles")
        .select("id,link")
        .eq("feed_id", FEED_ID)
        .execute()
        .data
    )

    if not articles:
        print("ℹ️ Aucun article pour ce feed")
        return 0

    article_ids = [a["id"] for a in articles]

    # 2) Links déjà scrapés
    existing_rows = (
        supabase
        .table("article_texts")
        .select("article_id")
        .in_("article_id", article_ids)
        .execute()
        .data
    )
    existing_ids = {r["article_id"] for r in existing_rows}

    # 3) À scraper
    todo = [a for a in articles if a["id"] not in existing_ids and a.get("link")]

    print(
        f"Feed {FEED_ID} | total={len(articles)} "
        f"| déjà={len(existing_ids)} | à_scraper={len(todo)}"
    )

    if not todo:
        print("✅ Rien à scraper")
        return 0

    payload = []

    with requests.Session() as session:
        for i, row in enumerate(todo, start=1):
            link = row["link"]
            article_id = row["id"]

            try:
                content, error = scrape_one(session, link)
            except Exception as e:
                print(f"[{i}/{len(todo)}] FAIL - {link} (exception: {e})")
                continue

            if error:
                print(f"[{i}/{len(todo)}] FAIL - {link} ({error})")
                continue

            payload.append({
                "article_id": article_id,
                "link": link,
                "content": content.split("\nÀ regarder\n-")[0],
            })

            print(f"[{i}/{len(todo)}] OK - {link}")
            time.sleep(1)  # simple throttling

    if payload:
        supabase.table("article_texts").upsert(
            payload,
            on_conflict="link"
        ).execute()
        print(f"✅ Upsert terminé : {len(payload)} lignes")
    else:
        print("ℹ️ Aucun contenu valide à insérer")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())