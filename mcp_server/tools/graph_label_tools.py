import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from supabase import create_client

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import pipeline
import torch

clf = pipeline(
    "text-classification",
    model="best_model_news",
    tokenizer="best_model_news",
    device=0 if torch.cuda.is_available() else -1
)

# --- Supabase ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)


def recup_traitement_bdd() -> pd.DataFrame:
    rows = []
    page_size = 1000
    start = 0

    while True:
        resp = (
            supabase
            .table("articles")
            .select("id, link, published_at, feed_id, title, summary, label")
            .range(start, start + page_size - 1)
            .execute()
        )

        batch = resp.data or []
        rows.extend(batch)

        if len(batch) < page_size:
            break
        start += page_size

    return pd.DataFrame(rows)



def register(mcp) -> None:
    @mcp.tool()
    def graph_label(top_n: int = 20) -> dict:
        """
        Génère un graphique (barres horizontales) sur les labels,
           l'enregistre dans ./outputs et renvoie le chemin du fichier.
        """

        df = recup_traitement_bdd()
        if df.empty:
            return {"type": "error", "error": "DataFrame vide."}
        if "label" not in df.columns:
            return {"type": "error", "error": "Colonne 'label' absente."}
        if "id" not in df.columns:
            return {"type": "error", "error": "Colonne 'id' absente. Ajoute 'id' dans le select()."}

        updated = 0
        failed = 0

        # --- 1) Labellisation + update direct ---
        s = df["label"]
        mask = s.isna() | (s.astype(str).str.strip().isin(["", "NA", "None", "null"]))

        df_na = df.loc[mask].copy()
        if not df_na.empty:
            # texte = title + summary
            texts = (
                df_na["title"].fillna("").astype(str).str.strip()
                + " "
                + df_na["summary"].fillna("").astype(str).str.strip()
            ).str.strip()

            # on ignore les textes vides
            non_empty = texts.str.len() > 0
            df_na = df_na.loc[non_empty]
            texts = texts.loc[non_empty].tolist()

            if len(texts) > 0:
                preds = clf(texts, truncation=True, top_k=3)
                pred_labels = [p[0]["label"] for p in preds]

                # update supabase directement
                for article_id, lab in zip(df_na["id"].tolist(), pred_labels):
                    try:
                        supabase.table("articles").update({"label": lab}).eq("id", article_id).execute()
                        updated += 1
                    except Exception:
                        failed += 1

                # recharger df pour que le dashboard reflète les updates
                df = recup_traitement_bdd()

        # --- 2) Dashboard ---
        counts = df["label"].fillna("NA").value_counts().head(int(top_n))

        project_root = Path(__file__).resolve().parent
        out_dir = project_root / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"dashboard_top{int(top_n)}_{ts}.png"

        fig, ax = plt.subplots(figsize=(10, 6))

        # --- Fond ---
        fig.patch.set_facecolor("#0e1117")   # fond figure (noir bleuté)
        ax.set_facecolor("#0e1117")          # fond axes

        # --- Barres ---
        bars = ax.barh(
            counts.index[::-1],
            counts.values[::-1],
            color="#4cc9f0"                  # bleu moderne
    )

        # --- Titres & labels ---
        ax.set_title(
            f"Répartition des catégories",
            fontsize=14,
            color="white",
            pad=15,
            weight="bold"
        )
        ax.set_xlabel("Nombre d'occurrences", color="white", labelpad=10)
        ax.set_ylabel("Catégorie", color="white", labelpad=10)

        # --- Ticks ---
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")

        # --- Grille légère ---
        ax.grid(
            axis="x",
            color="white",
            alpha=0.15,
            linestyle="--",
            linewidth=0.7
        )
        ax.set_axisbelow(True)

        # --- Suppression des spines ---
        for spine in ax.spines.values():
            spine.set_visible(False)

        # --- Valeurs au bout des barres ---
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + max(counts.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                va="center",
                ha="left",
                color="white",
                fontsize=10
            )

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return {
            "type": "image_saved",
            "path": str(out_path),
            "filename": out_path.name,
            "labellisation": {
                "updated": updated,
                "failed": failed,
            }
        }
