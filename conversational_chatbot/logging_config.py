import os
import logging
from logging.handlers import TimedRotatingFileHandler

# Configure logging
log_file_path = "logs/chatbot.log"  # Path to the log file
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure the logs directory exists

# Set up a TimedRotatingFileHandler to rotate logs daily
handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=7)
handler.suffix = "%Y-%m-%d"  # Add a date suffix to log files
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Configure the root logger
logging.basicConfig(level=logging.INFO, handlers=[handler])

# Example log message
logging.info("Logging is configured. Logs will be rotated daily.")
logging.info("Log file path: %s", log_file_path)