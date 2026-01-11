import os
import feedparser
from dateutil import parser as dateparser
from supabase import create_client

def to_iso_datetime(entry) -> str | None:
    for k in ("published", "updated", "pubDate"):
        v = entry.get(k)
        if v:
            try:
                return dateparser.parse(v).isoformat()
            except Exception:
                pass
    return None

def main():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]  # idéalement service_role pour un job serveur
    supabase = create_client(url, key)

    sources = supabase.table("feeds").select("*").execute().data

    for src in sources:
        feed = feedparser.parse(src["url"])

        rows = []
        for e in feed.entries:
            link = e.get("link")
            if not link:
                continue

            rows.append({
                "feed_id": src["id"],
                "title": e.get("title"),
                "link": link,
                "published_at": to_iso_datetime(e),
                "summary": e.get("summary") or e.get("description"),
            })

        if not rows:
            print(f"[{src['name']}] Aucun article trouvé.")
            continue

        # dédup dans le batch
        rows = list({r["link"]: r for r in rows}.values())

        supabase.table("articles").upsert(
            rows,
            on_conflict="link",
            ignore_duplicates=True
        ).execute()

        print(f"[{src['name']}] OK - {len(rows)} items upsertés.")

if __name__ == "__main__":
    main()
