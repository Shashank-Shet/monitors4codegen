"""
Multilspy logger module.
"""
import inspect
import logging

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

    def log(self, debug_message: str, level: int) -> None:
        """
        Log the messages using the logger
        """

        debug_message = debug_message.replace("'", '"').replace("\n", " ")

        # Collect details about the callee
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        caller_file = calframe[1][1].split("/")[-1]
        caller_line = calframe[1][2]
        # caller_name = calframe[1][3]

        msg = f"{caller_file}:{caller_line} - {debug_message}"

        # By default the log method of LoggerAdapter, checks if the given event
        # is allowed by the set level.
        super().log(level, msg)