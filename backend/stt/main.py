"""Point d'entree du STT : transcrit un fichier audio passe en argument.

Brique de la v1, volontairement non branchee dans l'API de la v0.
"""
import sys
import textwrap

from backend.stt.model import transcribe


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m backend.stt.main <fichier audio>", file=sys.stderr)
        return 2
    print(textwrap.fill(transcribe(sys.argv[1]), width=100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
