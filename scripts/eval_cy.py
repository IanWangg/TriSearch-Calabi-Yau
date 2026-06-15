import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.evaluate_rl_cy import main, parse_args
if __name__ == "__main__":
    main(parse_args())
