import argparse
from pathlib import Path
from sf_visuals.analyser import Analyser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path)
    args = parser.parse_args()

    analyser = Analyser(path=args.path)

    classes = {}
    for k in analyser.classes:
        classes[k] = input(f"Name for Class {k}: ")

    analyser.classes = classes

    print(analyser.classes)
