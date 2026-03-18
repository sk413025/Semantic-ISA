"""
Allow running eval as:
  python -m asir.eval                  # L4-L7 semantic eval (inject features)
  python -m asir.eval --integration    # End-to-end with real audio
  python -m asir.eval --compare A B    # A/B comparison (v0.9)

Flags (combinable):
  --verbose / -v         Show full L4→L5→L6 trace for every scenario
  --program <path>       Load optimized program from saved JSON
"""
import sys

if "--compare" in sys.argv:
    from asir.eval.compare import main
elif "--integration" in sys.argv:
    from asir.eval.integration import main
else:
    from asir.eval.run import main
main()
