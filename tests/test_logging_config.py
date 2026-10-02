import logging
import unittest

from libro_tts.logging_config import configure_logging


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class LoggingConfigTests(unittest.TestCase):
    def test_spark_transformers_warning_is_filtered(self):
        configure_logging(verbose=False)
        logger = logging.getLogger("transformers.configuration_utils")
        handler = _ListHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        try:
            logger.warning(
                "You are using a model of type spark to instantiate a model of type . "
                "This is not supported for all configurations of models and can yield errors."
            )
            logger.warning("unrelated warning")
        finally:
            logger.removeHandler(handler)

        self.assertEqual(handler.messages, ["unrelated warning"])


if __name__ == "__main__":
    unittest.main()
