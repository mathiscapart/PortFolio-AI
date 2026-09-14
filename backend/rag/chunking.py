"""Découpage de documents Markdown en chunks pour l'ingestion RAG.

Fonctions pures et déterministes : mêmes entrées -> mêmes sorties, y compris
l'ordre des chunks.
"""
from __future__ import annotations

import re

# Approximation du nombre de tokens : le service n'embarque pas de tokenizer
# dédié, un mot ~ un token pour du texte français/anglais courant. Suffisant
# pour dimensionner des chunks, pas pour une facturation.
def _count_tokens(text: str) -> int:
    return len(text.split())


_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Extrait un front-matter minimal (`cle: valeur` par ligne) en tête du document.

    Retourne le dict des clés trouvées et le corps du document, front-matter retiré.
    """
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    front_matter = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        cle, _, valeur = line.partition(":")
        front_matter[cle.strip()] = valeur.strip()
    return front_matter, text[match.end():]


def _split_sections(body: str) -> list[tuple[str | None, str | None, str, bool]]:
    """Découpe le corps par titres Markdown.

    Retourne des quadruplets `(titre, fil_ariane, texte, est_conteneur)`.
    `fil_ariane` est le chemin des titres parents jusqu'à ce titre inclus
    (« Parcours > Expériences professionnelles ») : sans lui, le libellé d'un
    conteneur disparaît du corpus entier puisqu'aucun chunk enfant ne le
    reprend, alors que ce sont ses mots (« expériences professionnelles »)
    que reprend la question d'un visiteur. `est_conteneur` vaut True quand le
    titre suivant est d'un niveau plus profond : la section n'a alors pas de
    contenu propre, elle ne fait qu'en chapeauter d'autres. Le texte précédant
    le premier titre, s'il existe, a un titre et un fil d'Ariane `None`.
    """
    headings = list(_HEADING_RE.finditer(body))
    if not headings:
        return [(None, None, body, False)]

    sections = []
    preambule = body[: headings[0].start()]
    if preambule.strip():
        sections.append((None, None, preambule, False))

    pile = []  # [(niveau, titre)] des ancêtres du titre courant
    for i, heading in enumerate(headings):
        titre = heading.group(2).strip()
        niveau = len(heading.group(1))
        while pile and pile[-1][0] >= niveau:
            pile.pop()
        pile.append((niveau, titre))
        fil_ariane = " > ".join(t for _, t in pile)

        debut = heading.end()
        suivant = headings[i + 1] if i + 1 < len(headings) else None
        fin = suivant.start() if suivant else len(body)
        est_conteneur = suivant is not None and len(suivant.group(1)) > niveau
        sections.append((titre, fil_ariane, body[debut:fin], est_conteneur))

    return sections


def _split_long_section(texte: str, max_tokens: int, overlap: int) -> list[str]:
    """Replie une section trop longue en fenêtres de mots avec recouvrement."""
    mots = texte.split()
    if len(mots) <= max_tokens:
        texte = texte.strip()
        return [texte] if texte else []

    pas = max_tokens - overlap
    morceaux = []
    debut = 0
    while debut < len(mots):
        morceaux.append(" ".join(mots[debut:debut + max_tokens]))
        debut += pas
    return morceaux


def chunk_markdown(
    text: str,
    source: str,
    max_tokens: int = 600,
    overlap: int = 80,
) -> list[dict]:
    """Découpe un document Markdown en chunks pour l'ingestion RAG.

    Découpage principal par titre Markdown ; repli en fenêtres de ~`max_tokens`
    mots avec un recouvrement de `overlap` mots quand une section dépasse ce
    seuil. Un front-matter `titre` / `source` en tête de document, s'il existe,
    prévaut sur les paramètres.

    Chaque chunk retourné porte : `texte` (titre de section inclus), `source`,
    `titre` (section d'origine, peut être `None`) et `index` (position).

    Fonction pure : aucun effet de bord, aucune I/O.
    """
    # Sans normalisation, un document en CRLF (core.autocrlf sous Windows) ne
    # matche pas le front-matter : `source` retombe sur le paramètre et tous les
    # `index` sont décalés, donc tous les uuid5 dérivés changent.
    text = text.replace("\r\n", "\n")

    # Avec overlap >= max_tokens, le pas de fenêtre vaut 0 : le découpage
    # n'avance jamais et boucle jusqu'à l'OOM. On échoue vite et clairement.
    if overlap >= max_tokens:
        raise ValueError(
            f"overlap ({overlap}) doit être strictement inférieur à "
            f"max_tokens ({max_tokens}) : sinon le découpage n'avance jamais."
        )

    front_matter, body = parse_front_matter(text)
    source = front_matter.get("source", source)
    titre_defaut = front_matter.get("titre")

    chunks = []
    index = 0
    for titre_section, fil_ariane, texte_section, est_conteneur in _split_sections(body):
        titre = titre_section or titre_defaut
        fil = fil_ariane or titre_defaut
        morceaux = _split_long_section(texte_section, max_tokens, overlap)

        # Une section titrée sans corps produisait zéro chunk : son titre
        # disparaissait en silence du corpus (« Contact », « Langues »).
        # Un titre sans corps ne donne un chunk que s'il s'agit d'une feuille
        # (« Contact », « Langues »). Un titre qui ne fait que chapeauter des
        # sous-sections produirait un chunk d'un ou deux mots, dont l'embedding
        # score haut sur n'importe quelle question courte et noie le vrai contenu.
        # `titre_section` et non `titre` : un `titre` de front-matter applique a un
        # corps vide produirait un chunk fantome reduit a ce seul mot, et un
        # document vide de son contenu se reindexerait en bruit au lieu d'etre purge.
        if not morceaux and titre_section and not est_conteneur:
            morceaux = [""]

        for morceau in morceaux:
            # Le fil d'Ariane (titres des parents jusqu'à celui-ci) doit
            # figurer dans le texte embedé, pas seulement le titre propre de
            # la section : sinon le libellé d'un conteneur (« Expériences
            # professionnelles ») disparaît de tout le corpus, aucun chunk
            # enfant ne le reprenant, et une question qui le cite ne matche
            # aucun vecteur.
            texte = f"{fil}\n\n{morceau}".strip() if fil else morceau
            chunks.append({
                "texte": texte,
                "source": source,
                "titre": titre,
                "index": index,
            })
            index += 1

    return chunks
