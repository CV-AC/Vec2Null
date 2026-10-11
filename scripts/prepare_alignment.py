import argparse
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method.unoface.sources import COMMIT, SOURCE_PATHS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="external/insightface")
    args = parser.parse_args()
    root = Path(args.output)
    upstream = root / "upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCE_PATHS.items():
        path = upstream / f"{name}.py"
        if path.is_file():
            continue
        url = f"https://raw.githubusercontent.com/deepinsight/insightface/{COMMIT}/{source}"
        with urlopen(url, timeout=60) as response:
            contents = response.read()
        path.write_bytes(contents)
    url = (
        f"https://raw.githubusercontent.com/deepinsight/insightface/{COMMIT}/README.md"
    )
    with urlopen(url, timeout=60) as response:
        (root / "README.upstream.md").write_bytes(response.read())
    print(f"Alignment sources prepared in {root}")


if __name__ == "__main__":
    main()
