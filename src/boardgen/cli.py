import argparse

from . import __version__


def main():
    parser = argparse.ArgumentParser(description="BoardGen tabletop reconstruction tools")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()


if __name__ == "__main__":
    main()
