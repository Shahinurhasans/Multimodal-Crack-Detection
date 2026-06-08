# ===================================================================
# Local RAG knowledge base for concrete-crack inspection reports.
#
# Each entry is a short, self-contained note written by a (fictional)
# structural-inspection reference guide. At report time we embed the
# model's verdict as a query, retrieve the most relevant notes with
# TF-IDF + cosine similarity, and hand them to the LLM as grounding
# context so the generated report stays consistent with real
# inspection terminology instead of improvising from scratch.
# ===================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS = [
    "Hairline cracks are surface cracks narrower than about 0.3 mm. They "
    "typically form from concrete shrinkage during curing and rarely "
    "indicate a structural problem, but should still be logged and "
    "monitored for growth over time.",

    "Cracks wider than 3 mm, or cracks that are diagonal, stepped, or "
    "running through structural members (beams, columns, load-bearing "
    "walls), are considered high severity and warrant a follow-up "
    "inspection by a licensed structural engineer.",

    "Vertical and horizontal cracks in walls are often caused by drying "
    "shrinkage or thermal movement. Diagonal cracks radiating from the "
    "corners of openings (doors, windows) frequently indicate "
    "foundation settlement or differential movement of the structure.",

    "Map cracking (a network of fine interconnected surface cracks) is "
    "usually caused by alkali-silica reaction (ASR), poor curing, or "
    "surface drying too quickly after placement. It is mainly a "
    "durability concern rather than an immediate structural risk.",

    "Active cracks continue to widen or lengthen over time and need "
    "monitoring with crack gauges or periodic photographs; dormant "
    "cracks have stabilized and mainly need sealing to keep moisture "
    "and chlorides out of the concrete.",

    "Acoustic (tap / knock) testing is a simple non-destructive method: "
    "a solid, sharp, ringing sound generally indicates dense, intact "
    "concrete, while a dull, hollow, or flat sound can indicate internal "
    "delamination, voids, or subsurface cracking that is not visible "
    "from the surface.",

    "Corrosion of embedded steel reinforcement is a common root cause of "
    "cracking in older concrete: rust expands to several times the "
    "volume of the original steel, and the resulting pressure splits the "
    "surrounding concrete, often producing cracks that follow the line "
    "of the rebar with rust staining at the surface.",

    "Freeze-thaw cycles force water that has seeped into surface pores "
    "and small cracks to expand as it freezes, gradually widening the "
    "cracks and spalling (flaking) the surface. This is most common in "
    "climates with cold winters and on horizontal or poorly-drained "
    "surfaces.",

    "Recommended remediation depends on severity: hairline and dormant "
    "cracks are usually sealed with a flexible polyurethane or epoxy "
    "sealant to block moisture; active or structural cracks may require "
    "epoxy injection, carbon-fiber reinforcement, or engineered repair "
    "designed by a structural engineer.",

    "When no visible cracking is detected, routine maintenance still "
    "matters: keep drainage paths clear so water does not pond on the "
    "surface, reapply protective sealants on the manufacturer's "
    "schedule, and re-inspect after extreme weather events (heavy "
    "freeze-thaw seasons, floods, earthquakes).",

    "A documented inspection record (photo plus written note plus date) "
    "is the single most useful tool for tracking whether a crack is "
    "growing: comparing today's image and measurements with the next "
    "inspection is what actually reveals whether a crack is active.",

    "General safety guidance: cracks accompanied by other warning signs "
    "-- sagging floors or roof lines, doors and windows that suddenly "
    "stick, leaning walls, or fresh water infiltration through the crack "
    "-- should be treated as urgent and assessed by a professional as "
    "soon as possible.",
]


_vectorizer = TfidfVectorizer(stop_words="english")
_doc_matrix = _vectorizer.fit_transform(DOCUMENTS)


def retrieve(query, top_k=4):
    """Return the `top_k` knowledge-base notes most relevant to `query`."""
    query_vector = _vectorizer.transform([query])
    scores = cosine_similarity(query_vector, _doc_matrix)[0]
    ranked = sorted(range(len(DOCUMENTS)), key=lambda i: scores[i], reverse=True)
    return [DOCUMENTS[i] for i in ranked[:top_k] if scores[i] > 0]
