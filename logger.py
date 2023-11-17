import logging
from logging.handlers import TimedRotatingFileHandler


debug_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s.%(funcName)s l.%(lineno)d | %(message)s')
debug_file_handler = TimedRotatingFileHandler(
    filename="features.log", when='midnight', backupCount=31)
debug_file_handler.setFormatter(debug_formatter)
debug_file_handler.setLevel(logging.DEBUG)

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(debug_formatter)

logger = logging.getLogger("features")
logger.setLevel(logging.DEBUG)
logger.addHandler(debug_file_handler)
logger.addHandler(console_handler)