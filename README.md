# 📰 News360 — Agent IA pour l’analyse intelligente de l’actualité

## Présentation générale

**News360** est un **chatbot intelligent orienté actualités**, basé sur un **agent LLM orchestrant des outils MCP (Model Context Protocol)**.  
Il permet d’interroger l’actualité en langage naturel, d’analyser des articles de presse à grande échelle et de produire des **synthèses, analyses thématiques, dashboards et indicateurs de sentiment**.

Le projet repose sur une architecture **agent + tools**, où le LLM agit comme un **chef d’orchestre** capable de :
- rechercher des articles (API + base interne),
- analyser leur contenu (labels, sentiment),
- résumer et reformuler,
- produire des statistiques et visualisations.

---

## Objectifs du projet

- Centraliser l’accès à l’actualité via plusieurs sources  
- Permettre une **analyse thématique et temporelle** des sujets d’actualité  
- Exploiter des **modèles NLP modernes** (LLM, Transformers, CamemBERT)  
- Illustrer une architecture **agentique** moderne (MCP + LangChain)  
- Automatiser l’ingestion et l’analyse quotidienne de données de presse  

---

## Architecture globale

### Architecture logique

```text
Utilisateur
   │
   ▼
Chat CLI (client.py)
   │
   ▼
Agent LLM (Groq / Ollama)
   │
   ▼
Serveur MCP (FastMCP)
   │
   ├── Recherche News (NewsAPI)
   ├── Recherche Base interne (DB)
   ├── Analyse de sentiment
   ├── Labellisation thématique
   ├── Dashboard (statistiques + graphiques)
   ├── Résumé d’articles
   └── PDF → Texte
```


---

## Fonctionnalités détaillées

### 1️. Recherche d’actualité via NewsAPI

**Fonctionnement :**
1. L’utilisateur pose une question (ex: *« Que dit la presse sur Trump ? »*)
2. L’agent détecte une intention *actualité*
3. Appel du tool MCP `search_news(topic="Trump")`
4. Récupération d’environ **25 articles récents**
5. Le LLM lit les résultats et produit une **synthèse argumentée**

---

### 2️. Recherche dans une base d’articles interne

**Objectif :** disposer d’un historique indépendant des API externes.

- Base alimentée **quotidiennement depuis le 24 décembre**
- Ingestion automatisée via **GitHub Actions**
- Requêtes possibles :
  - mot-clé (ex: *Trump*)
  - intervalle de dates (ex: *2 janvier → 5 janvier*)
- Recherche actuelle basée sur le **titre**
- Sortie : liste Python d’articles correspondants

---

### 3️. Labellisation thématique & Dashboard

#### Labels utilisés
- Politique  
- Économie  
- International  
- Société  
- Technologie  
- Environnement  
- Autres

---

#### Exemple de requête
> *« Quels sont les thèmes principalement abordés depuis le 1er janvier 2025 ? »*

---

### 4️. Analyse de sentiment

- Modèle Transformers (Hugging Face)
- Sortie : score positif / négatif
- Limite assumée :
  - les articles de presse sont souvent **neutres**
  - l’indicateur est utilisé comme **signal**, pas comme vérité absolue

---

### 5️. Résumé automatique d’articles

**Pipeline :**
1. Scraping de l’article à partir de l’URL
2. Résumé via un modèle **Facebook / Hugging Face**
3. Reformulation éventuelle par le LLM
4. Résumé clair, concis et contextualisé

---

### 6️. PDF → Texte

- Outil utilitaire
- Extraction du texte page par page
- Réutilisable pour :
  - analyse
  - résumé
  - classification

---

## Fonctionnement du MCP (Model Context Protocol)

- Les tools sont exposés par **FastMCP**
- Le LLM :
  - ne code pas
  - choisit dynamiquement le bon outil
  - fournit les paramètres adaptés
- Le serveur MCP exécute l’action
- Le LLM synthétise la réponse finale

**Principe clé :**
> Séparation stricte entre **raisonnement** et **exécution**

---

## Ingestion automatique des données


- Pipeline opérationnel depuis le **24 décembre**
- Base continuellement enrichie

---

## Lancer le projet

```bash
git clone https://github.com/ZinebMnr/NewsBot360.git
cd news360
pip install -r requirements_mcp.txt
python client.py
```

## Auteurs

Aymane Aibichi 
Zineb Manar
