"""Allow ``python -m embed_parity``."""

import sys

from embed_parity.cli import main

if __name__ == "__main__":
    sys.exit(main())
