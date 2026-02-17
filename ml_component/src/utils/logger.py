
import logging
import sys

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
        datefmt='%m-%d %H:%M',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def get_logger(name):
    return logging.getLogger(name)

if __name__ == '__main__':
    setup_logging()
    logger = get_logger("Test")
    logger.info("Test message")
