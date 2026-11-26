"""Makes the release importable without installation (adds the repo root and third_party/)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(ROOT, "third_party"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
