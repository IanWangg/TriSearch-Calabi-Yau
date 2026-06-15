import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    from core.train_cy import main, parse_args

    main(parse_args())
