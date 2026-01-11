import os
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from supabase import create_client
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

load_dotenv()

# Racine du projet (ex: D:\projets\NewsBot360)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS = [
    "POLITIQUE",
    "ECONOMIE",
    "ENTREPRISES",
    "TECHNOLOGIE",
    "SCIENCE_SANTE",
    "SOCIETE",
    "FAITS_DIVERS_JUSTICE",
    "SPORT",
    "CULTURE_LOISIRS",
    "AUTRE_INDETERMINE",
]

MODEL_NAME = "camembert-base"
OUTPUT_DIR = PROJECT_ROOT / "news_clf_camembert"
BEST_MODEL_DIR = PROJECT_ROOT / "best_model_news"

META_FILE = "training_meta.json"


def get_supabase_client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "Variables d'environnement manquantes : SUPABASE_URL et/ou SUPABASE_KEY."
        )
    return create_client(supabase_url, supabase_key)


def fetch_articles_dataframe(
    supabase,
    table_name: str = "articles",
    page_size: int = 1000,
) -> pd.DataFrame:
    rows: List[dict] = []
    start = 0

    while True:
        resp = (
            supabase.table(table_name)
            .select("title,summary,label")
            .range(start, start + page_size - 1)
            .not_.is_("label", "null")
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)

        if len(batch) < page_size:
            break

        start += page_size

    return pd.DataFrame(rows)


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def prepare_dataframe(df: pd.DataFrame, label2id: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["title"].fillna("") + " " + df["summary"].fillna("")
    df = df.dropna(subset=["text", "label"]).copy()

    df["label_id"] = df["label"].map(label2id)

    # drop labels inconnus
    df = df.dropna(subset=["label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)

    return df


def split_datasets(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dataset, Dataset]:
    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_id"],
    )

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]], preserve_index=False)
    valid_ds = Dataset.from_pandas(valid_df[["text", "label_id"]], preserve_index=False)
    return train_ds, valid_ds


def tokenize_datasets(
    train_ds: Dataset,
    valid_ds: Dataset,
    tokenizer,
    max_length: int = 256,
) -> Tuple[Dataset, Dataset]:
    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(preprocess, batched=True)
    valid_ds = valid_ds.map(preprocess, batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    valid_ds = valid_ds.rename_column("label_id", "labels")

    return train_ds, valid_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def build_training_args(output_dir: Path, num_train_epochs: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.6,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        save_safetensors=False,  # Windows-friendly
    )


def _load_total_epochs(best_model_dir: Path) -> int:
    meta_path = best_model_dir / META_FILE
    if meta_path.exists():
        try:
            return int(json.loads(meta_path.read_text(encoding="utf-8")).get("total_epochs", 0))
        except Exception:
            return 0
    return 0


def _save_total_epochs(best_model_dir: Path, total_epochs: int) -> None:
    best_model_dir.mkdir(parents=True, exist_ok=True)
    meta_path = best_model_dir / META_FILE
    meta_path.write_text(
        json.dumps({"total_epochs": int(total_epochs)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _safe_save_best_model(trainer: Trainer, tokenizer, best_model_dir: Path) -> None:
    """
    Sauvegarde robuste sur Windows:
    - test d'écriture
    - save dans un dossier tmp
    - remplacement du dossier final
    """
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # Test écriture simple (droits / lock)
    test_path = best_model_dir / ".__write_test__"
    try:
        test_path.write_text("ok", encoding="utf-8")
        try:
            test_path.unlink()
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError(
            f"Impossible d'écrire dans {best_model_dir}. "
            f"Cause probable: dossier verrouillé (Explorer/VSCode/AV/OneDrive) ou droits.\n"
            f"Détail: {repr(e)}"
        )

    tmp_dir = best_model_dir.parent / f"{best_model_dir.name}__tmp"

    # Nettoyage tmp
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde dans tmp
    trainer.save_model(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))

    # Petite pause pour laisser Windows/AV respirer
    time.sleep(0.5)

    # Remplacement
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir, ignore_errors=True)

    shutil.move(str(tmp_dir), str(best_model_dir))


def train_and_save(
    train_ds: Dataset,
    valid_ds: Dataset,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    model_name: str = MODEL_NAME,
    output_dir: Path = OUTPUT_DIR,
    best_model_dir: Path = BEST_MODEL_DIR,
):
    # Best existe seulement si config.json est là (modèle HF complet)
    best_exists = (best_model_dir / "config.json").exists()

    # Si best existe, on repart de best, sinon de camembert-base
    base_for_loading = str(best_model_dir) if best_exists else model_name

    tokenizer = AutoTokenizer.from_pretrained(base_for_loading)
    train_ds, valid_ds = tokenize_datasets(train_ds, valid_ds, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_for_loading,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # +1 epoch si best existe, sinon 3
    epochs_this_run = 1 if best_exists else 3
    args = build_training_args(output_dir, num_train_epochs=epochs_this_run)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,  # warning deprecation OK
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    # Sauvegarde robuste
    _safe_save_best_model(trainer, tokenizer, best_model_dir)

    prev_total = _load_total_epochs(best_model_dir) if best_exists else 0
    _save_total_epochs(best_model_dir, prev_total + epochs_this_run)

    return metrics


def main():
    label2id, id2label = build_label_maps(LABELS)

    supabase = get_supabase_client()
    df = fetch_articles_dataframe(supabase)

    df = prepare_dataframe(df, label2id)
    train_ds, valid_ds = split_datasets(df)

    train_and_save(
        train_ds=train_ds,
        valid_ds=valid_ds,
        label2id=label2id,
        id2label=id2label,
    )


if __name__ == "__main__":
    main()
