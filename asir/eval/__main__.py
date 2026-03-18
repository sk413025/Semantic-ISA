"""
python -m asir.eval                          # L4-L7 semantic eval
python -m asir.eval --integration            # End-to-end with real audio
python -m asir.eval --program <path>         # Load optimized program
"""
import sys

if "--integration" in sys.argv:
    from asir.eval.integration import main
else:
    from asir.eval.run import main
main()
