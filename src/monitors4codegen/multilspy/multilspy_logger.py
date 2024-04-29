"""
Multilspy logger module.
"""
import inspect
import logging

TRACE = 5

class MultilspyLogger(logging.LoggerAdapter):
    """ Custom Logger class for Multilspy.

    Subclasses logging.LoggerAdapter to reuse the debug, info, warning, error, and critical methods.
    Uses a custom formatting, and overrides the log method so every log event uses the custom format.
    """

    def __init__(self, level = logging.INFO, fh: logging.FileHandler = None) -> None:
        logger = logging.getLogger("multilspy")
        logger.setLevel(level)
        if fh is None:
            fh = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        super().__init__(logger, {})

    def setCustomLevels(self):
        logging.addLevelName(TRACE, "TRACE")

    def log(self, level: int, debug_message: str) -> None:
        """
        Log the messages using the logger,
        Do not use directly, use the debug, info, warning, error, and critical methods instead.
        """

        # debug_message = debug_message.replace("'", '"').replace("\n", " ")
        debug_message = debug_message.replace("'", '"')

        # Collect details about the callee
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        caller_file = calframe[2][1].split("/")[-1]
        caller_line = calframe[2][2]
        # caller_name = calframe[1][3]

        msg = f"{caller_file}:{caller_line} - {debug_message}"

        # By default the log method of LoggerAdapter, checks if the given event
        # is allowed by the set level.
        super().log(level, msg)

    def trace(self, debug_message: str) -> None:
        """
        Log a trace message.
        """
        self.log(TRACE, debug_message)