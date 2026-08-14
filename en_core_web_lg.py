def load(**kwargs):
    # Shim that redirects requests for en_core_web_lg to the smaller en_core_web_sm.
    # If the small model isn't installed, fall back to a lightweight blank English model
    # so downstream code (Presidio) can initialize without triggering heavy downloads.
    import spacy
    try:
        return spacy.load("en_core_web_sm", **kwargs)
    except Exception:
        return spacy.blank("en")

__version__ = "3.8.0-shim"
