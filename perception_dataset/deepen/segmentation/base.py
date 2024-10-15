from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from ..deepen_annotation import DeepenAnnotation


class DeepenSegmentation(ABC):
    def __init__(
        self, input_anno_file: str, input_base: str, dataset_corresponding: Dict[str, str]
    ) -> None:
        super().__init__()
        self.input_anno_file = Path(input_anno_file)
        self.input_base = Path(input_base)
        self.dataset_corresponding = dataset_corresponding

    @abstractmethod
    def to_deepen_annotations(self) -> List[DeepenAnnotation]:
        """Convert deepen annotation format to a list of `DeepenAnnotation` objects.

        Returns:
            List[DeepenAnnotation]: List of converted objects.
        """
        pass

    @abstractmethod
    def to_bbox(self, *args, **kwargs):
        pass


class DeepenSegmentation2D(DeepenSegmentation):
    @abstractmethod
    def _get_image_size(self) -> tuple[int, int]:
        """Return the image size.

        Returns:
            tuple[int, int]: Image size ordering in (width, height).
        """
        pass
