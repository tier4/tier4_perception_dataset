import importlib
from types import ModuleType
import warnings


def _try_import_aip_module(
    module_name: str, *, warn_message: str | None = None
) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, SystemError) as exc:
        if warn_message is not None:
            warnings.warn(f"{warn_message}: {exc}")

        return None


def try_import_aip_common(*, warn: bool = False) -> ModuleType | None:
    """Try to import the accelerated_image_processor common module.

    Args:
        warn (bool, optional): Emit a warning when the common module cannot be imported.

    Returns:
        ModuleType | None: The common module, or `None` when it is unavailable.
    """
    warn_message = None
    if warn:
        warn_message = "Cannot import accelerated_image_processor.common module."

    return _try_import_aip_module("accelerated_image_processor.common", warn_message=warn_message)


def try_import_aip_decompression(*, warn: bool = False) -> ModuleType | None:
    """Try to import the accelerated_image_processor decompression module.

    Args:
        warn (bool, optional): Emit a warning when the decompression module cannot be imported.

    Returns:
        ModuleType | None: The decompression module, or `None` when it is unavailable.
    """
    warn_message = None
    if warn:
        warn_message = "Cannot import accelerated_image_processor.decompression module."

    return _try_import_aip_module(
        "accelerated_image_processor.decompression", warn_message=warn_message
    )
