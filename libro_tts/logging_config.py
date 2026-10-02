from __future__ import annotations

import logging
import warnings


_SPARK_TRANSFORMERS_WARNING_FRAGMENT = (
    "You are using a model of type spark to instantiate a model of type ."
)


class _SparkTransformersConfigWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return _SPARK_TRANSFORMERS_WARNING_FRAGMENT not in message


def _suppress_known_upstream_warnings() -> None:
    transformers_logger = logging.getLogger("transformers.configuration_utils")
    if not any(
        isinstance(existing_filter, _SparkTransformersConfigWarningFilter)
        for existing_filter in transformers_logger.filters
    ):
        transformers_logger.addFilter(_SparkTransformersConfigWarningFilter())

    # mlx-lm currently emits this deprecation warning from dependency code paths.
    warnings.filterwarnings(
        "ignore",
        message=(
            r"mx\.metal\.device_info is deprecated and will be removed in a future version\. "
            r"Use mx\.device_info instead\."
        ),
        category=Warning,
    )


def configure_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
    _suppress_known_upstream_warnings()
    return logging.getLogger("libro_tts")
