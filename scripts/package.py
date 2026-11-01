"""Build a portable source archive, excluding data, weights and local state."""
import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"README.md", "pyproject.toml", "requirements.txt", "requirements-lock.txt", ".gitignore"}
SOURCE_DIRS = {"madp", "configs", "docs", "tests", "scripts"}
SOURCE_SUFFIXES = {".py", ".json", ".md", ".txt"}


def package(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_symlink() or not path.is_file() or path.resolve() == destination:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts[:-1]):
            continue
        if (len(relative.parts) == 1 and path.name in TOP_FILES) or (
            relative.parts[0] in SOURCE_DIRS and path.suffix in SOURCE_SUFFIXES
        ):
            files.append(path)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, Path("madp-paper") / path.relative_to(ROOT))
    print(f"Packaged {len(files)} source files: {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist" / "madp-paper.zip"))
    package(parser.parse_args().output)
