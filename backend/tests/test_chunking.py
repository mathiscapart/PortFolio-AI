"""Tests du découpage Markdown (`backend/rag/chunking.py`).

Fonction pure : pas de dépendance à Qdrant ni Ollama.
"""
from pathlib import Path

import pytest

from backend.rag.chunking import chunk_markdown

FIXTURES = Path(__file__).parent / "fixtures"


def _lire(nom: str) -> str:
    return (FIXTURES / nom).read_text(encoding="utf-8")


def test_determinisme_meme_entree_meme_sortie():
    """Deux appels sur le même texte produisent une liste de chunks identique, ordre compris."""
    texte = _lire("fixture_animaux.md")
    premiere_passe = chunk_markdown(texte, source="fixture_animaux.md")
    deuxieme_passe = chunk_markdown(texte, source="fixture_animaux.md")
    assert premiere_passe == deuxieme_passe


def test_index_reparti_depuis_zero_par_document():
    """`index` repart de 0 pour chaque document découpé indépendamment."""
    for nom in ("fixture_animaux.md", "fixture_recettes.md", "fixture_terrain.md"):
        chunks = chunk_markdown(_lire(nom), source=nom)
        assert [c["index"] for c in chunks] == list(range(len(chunks)))


def test_total_21_chunks_sur_les_trois_fixtures():
    """Verrou de non-régression sur le nombre total de chunks produits par le corpus de fixtures."""
    total = sum(
        len(chunk_markdown(_lire(nom), source=nom))
        for nom in ("fixture_animaux.md", "fixture_recettes.md", "fixture_terrain.md")
    )
    assert total == 21


def test_front_matter_prend_le_pas_sur_les_parametres():
    """`titre`/`source` du front-matter l'emportent sur les paramètres passés en argument."""
    texte = _lire("fixture_recettes.md")
    chunks = chunk_markdown(texte, source="parametre_ignore.md")
    assert all(c["source"] == "fixture_recettes.md" for c in chunks)


def test_front_matter_titre_par_defaut_sans_titre_de_section():
    """Le `titre` du front-matter s'applique au texte qui précède le premier titre Markdown."""
    texte = (
        "---\n"
        "titre: Titre par defaut\n"
        "source: doc.md\n"
        "---\n"
        "Un paragraphe sans titre de section.\n"
    )
    chunks = chunk_markdown(texte, source="ignore.md")
    assert len(chunks) == 1
    assert chunks[0]["titre"] == "Titre par defaut"
    assert chunks[0]["source"] == "doc.md"


def test_recouvrement_section_longue_sans_perte_de_contenu():
    """Sur la section de 780 mots, les fenêtres se recouvrent et aucun mot n'est perdu."""
    texte = _lire("fixture_terrain.md")
    chunks = chunk_markdown(texte, source="fixture_terrain.md")

    # La première section ("Le plateau de Vorenn") dépasse max_tokens=600 : elle
    # doit être repliée en plusieurs fenêtres, contrairement à la seconde section.
    section_repliee = [c for c in chunks if c["titre"] == "Le plateau de Vorenn"]
    assert len(section_repliee) >= 2

    # Le titre est désormais préfixé à chaque fenêtre (cf. correctif #4) : on
    # compare le recouvrement sur le corps de la fenêtre, titre exclu, sinon
    # on compare les mots du titre entre eux et l'assertion ne verrouille rien.
    titre_mots = "Le plateau de Vorenn".split()

    def _corps(texte_chunk: str) -> list[str]:
        mots = texte_chunk.split()
        assert mots[: len(titre_mots)] == titre_mots
        return mots[len(titre_mots):]

    for precedent, suivant in zip(section_repliee, section_repliee[1:]):
        mots_precedent = _corps(precedent["texte"])
        mots_suivant = _corps(suivant["texte"])
        recouvrement = mots_precedent[-80:]
        assert mots_suivant[: len(recouvrement)] == recouvrement

    # Aucun mot du texte source n'est perdu entre les fenêtres : la dernière fenêtre
    # doit couvrir le dernier mot de la section d'origine.
    mots_source = texte.split("# Le plateau de Vorenn")[1].split("# La faille de Kastel")[0].split()
    assert _corps(section_repliee[-1]["texte"])[-1] == mots_source[-1]


def test_section_courte_non_repliee():
    """Une section sous le seuil `max_tokens` reste un seul chunk (pas de repli inutile)."""
    texte = _lire("fixture_terrain.md")
    chunks = chunk_markdown(texte, source="fixture_terrain.md")
    section_courte = [c for c in chunks if c["titre"] == "La faille de Kastel"]
    assert len(section_courte) == 1


def test_crlf_produit_les_memes_chunks_que_lf():
    """Un document en CRLF (core.autocrlf sous Windows) produit des chunks
    strictement identiques à sa version LF : sans normalisation, le
    front-matter ne matchait plus, `source` retombait sur le paramètre
    d'appel et tous les `index` étaient décalés d'une unité."""
    texte_lf = (
        "---\n"
        "titre: Titre par defaut\n"
        "source: doc.md\n"
        "---\n"
        "# Section\n"
        "\n"
        "Un paragraphe de contenu.\n"
    )
    texte_crlf = texte_lf.replace("\n", "\r\n")

    chunks_lf = chunk_markdown(texte_lf, source="ignore.md")
    chunks_crlf = chunk_markdown(texte_crlf, source="ignore.md")

    assert chunks_lf == chunks_crlf


def test_titre_de_section_present_dans_le_texte_embede():
    """Le titre de section doit figurer dans `texte` (pas seulement dans `titre`) :
    sinon une question sur ce titre ne peut matcher aucun vecteur."""
    texte = "# Experience professionnelle\n\nQuelques annees en tant qu'ingenieur.\n"
    chunks = chunk_markdown(texte, source="doc.md")
    assert len(chunks) == 1
    assert "Experience professionnelle" in chunks[0]["texte"]


def test_section_titree_sans_corps_produit_un_chunk():
    """Une section titrée sans texte (« Contact », « Langues ») produisait zéro
    chunk auparavant : son titre disparaissait en silence du corpus."""
    texte = "# Contact\n\n# Langues\n\nFrancais, anglais.\n"
    chunks = chunk_markdown(texte, source="doc.md")
    titres = [c["titre"] for c in chunks]
    assert "Contact" in titres

    chunk_contact = next(c for c in chunks if c["titre"] == "Contact")
    assert chunk_contact["texte"] == "Contact"


def test_titre_conteneur_sans_corps_ne_produit_aucun_chunk():
    """Un titre qui ne fait que chapeauter une sous-section (« Contact » suivi
    d'un `##`) ne doit produire aucun chunk portant son propre titre : avant la
    règle sur `est_conteneur`, ce cas produisait un chunk réduit à ce seul mot."""
    texte = "# Contact\n\n## Email\n"
    chunks = chunk_markdown(texte, source="doc.md")
    titres = [c["titre"] for c in chunks]
    assert "Contact" not in titres
    assert titres == ["Email"]


def test_titre_conteneur_avec_sous_section_a_corps_ne_produit_que_le_chunk_de_la_feuille():
    """Un titre conteneur suivi d'une sous-section avec du texte (« Parcours »
    puis « Qui je suis ») ne doit donner qu'un chunk : celui de la feuille."""
    texte = "# Parcours\n\n## Qui je suis\n\nTexte.\n"
    chunks = chunk_markdown(texte, source="doc.md")
    titres = [c["titre"] for c in chunks]
    assert "Parcours" not in titres
    assert titres == ["Qui je suis"]


def test_feuille_imbriquee_sans_corps_garde_son_chunk():
    """Une feuille sans corps imbriquée sous un titre qui a du contenu (`##`
    avec texte, puis `###` vide) reste un chunk : seuls les conteneurs sans
    corps propre doivent disparaître, pas toute section sans texte."""
    texte = "## Section\n\nDu texte.\n\n### Sous-section\n"
    chunks = chunk_markdown(texte, source="doc.md")
    titres = [c["titre"] for c in chunks]
    assert titres == ["Section", "Sous-section"]
    chunk_sous_section = next(c for c in chunks if c["titre"] == "Sous-section")
    assert chunk_sous_section["texte"] == "Section > Sous-section"


def test_fil_ariane_des_ancetres_dans_le_texte_embede():
    """Le texte embedé d'un chunk profond porte les titres de tous ses
    ancêtres, pas seulement le sien : sans ça, le libellé d'un conteneur
    (« Expériences professionnelles ») disparaît de tout le corpus, aucun
    chunk enfant ne le reprenant, et une question qui le cite ne matche
    aucun vecteur (finding 3)."""
    texte = (
        "# Parcours\n\n"
        "## Expériences professionnelles\n\n"
        "### Ingénieure IA — Aubelis\n\n"
        "Détail du poste.\n"
    )
    chunks = chunk_markdown(texte, source="doc.md")
    chunk_feuille = next(c for c in chunks if c["titre"] == "Ingénieure IA — Aubelis")
    for ancetre in ("Parcours", "Expériences professionnelles", "Ingénieure IA — Aubelis"):
        assert ancetre in chunk_feuille["texte"]


def test_aucun_titre_du_document_absent_du_corpus_concatene():
    """Propriété métier verrouillée par le finding 3 : sur un document réaliste
    à trois niveaux, aucun titre présent dans le document n'est absent du texte
    concaténé de ses chunks. Avant le fil d'Ariane, les titres des conteneurs
    (« Parcours », « Expériences professionnelles ») disparaissaient en silence."""
    texte = (
        "# Parcours\n\n"
        "## Expériences professionnelles\n\n"
        "### Ingénieure IA — Aubelis\n\n"
        "Détail du poste.\n\n"
        "## Formation\n\n"
        "### Master informatique\n\n"
        "Détail de la formation.\n"
    )
    chunks = chunk_markdown(texte, source="doc.md")
    corpus_concatene = "\n".join(c["texte"] for c in chunks)
    titres_du_document = [
        "Parcours",
        "Expériences professionnelles",
        "Ingénieure IA — Aubelis",
        "Formation",
        "Master informatique",
    ]
    for titre in titres_du_document:
        assert titre in corpus_concatene


def test_titre_du_payload_reste_le_titre_de_section_seul():
    """Le champ `titre` du payload est le titre de la section elle-même, pas
    le fil d'Ariane : ce sont deux informations distinctes, l'une sert de
    métadonnée de filtrage, l'autre enrichit le texte embedé."""
    texte = "# Parcours\n\n## Expériences professionnelles\n\n### Stage\n\nDétail.\n"
    chunks = chunk_markdown(texte, source="doc.md")
    chunk_feuille = next(c for c in chunks if c["titre"] == "Stage")
    assert chunk_feuille["titre"] == "Stage"
    assert " > " not in chunk_feuille["titre"]


def test_hierarchie_a_trois_niveaux_ne_garde_que_la_feuille():
    """h1 conteneur -> h2 conteneur -> h3 avec corps : seul le chunk du h3
    doit être produit, les deux conteneurs intermédiaires disparaissent."""
    texte = "# Experiences\n\n## Ingenieur\n\n### Stage R&D\n\nDetail du stage.\n"
    chunks = chunk_markdown(texte, source="doc.md")
    titres = [c["titre"] for c in chunks]
    assert titres == ["Stage R&D"]


def test_overlap_superieur_ou_egal_a_max_tokens_leve_value_error():
    """Avec overlap >= max_tokens, le pas de fenêtre vaut 0 et le découpage
    n'avance jamais (boucle jusqu'à l'OOM) : on doit échouer vite et clairement."""
    texte = "# Titre\n\n" + " ".join(f"mot{i}" for i in range(20))
    with pytest.raises(ValueError):
        chunk_markdown(texte, source="doc.md", max_tokens=10, overlap=10)
