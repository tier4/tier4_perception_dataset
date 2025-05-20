import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple


class DuplicatedAnnotationRemover:
    def _find_annotation_files(
        self,
        root_dir: str,
        target_filenames: Tuple[str, ...] = ("object_ann.json", "surface_ann.json"),
    ) -> List[Path]:
        """Recursively search for annotation files under the specified folder."""
        ann_files: List[Path] = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename in target_filenames:
                    ann_files.append(Path(dirpath) / filename)
        return ann_files

    def _deduplicate_annotations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entries where information other than the token is identical."""
        seen: Set[str] = set()
        unique: List[Dict[str, Any]] = []
        for entry in data:
            # Use the content excluding the token as a comparison key
            key = json.dumps({k: v for k, v in entry.items() if k != "token"}, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        return unique

    def _process_annotation_file(self, file_path: Path) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)

        before = len(data)
        data = self._deduplicate_annotations(data)
        after = len(data)

        print(f"{file_path}: {before} → {after} entries after deduplication")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def remove_duplicated_annotation(self, root_dir: str) -> None:
        ann_files = self._find_annotation_files(root_dir)
        if not ann_files:
            print("No annotation files were found.")
            return
        for file_path in ann_files:
            self._process_annotation_file(file_path)


if __name__ == "__main__":
    remover = DuplicatedAnnotationRemover()

    if len(sys.argv) != 2:
        print("Usage: python remove_duplicates.py <directory_path>")
    else:
        remover.remove_duplicated_annotation(sys.argv[1])
