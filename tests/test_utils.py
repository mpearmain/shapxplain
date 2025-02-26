import pytest
import logging
from io import StringIO
from shapxplain.utils import setup_logger, preprocess_data, logger

def test_setup_logger():
    """Test the logger setup function."""
    # Test with default parameters
    test_logger = setup_logger()
    assert test_logger.level == logging.INFO
    assert len(test_logger.handlers) > 0
    
    # Test with custom level
    test_logger = setup_logger(level=logging.DEBUG)
    assert test_logger.level == logging.DEBUG
    
    # Test with custom format
    custom_format = "%(levelname)s - %(message)s"
    test_logger = setup_logger(log_format=custom_format)
    assert test_logger.handlers[0].formatter._fmt == custom_format

def test_preprocess_data():
    """Test the preprocess_data function."""
    # Simple test data
    test_data = {"key": "value"}
    
    # Should return data unchanged since it's a placeholder function
    result = preprocess_data(test_data)
    assert result == test_data

def test_logger_output():
    """Test the logging output using a string buffer."""
    # Create a string buffer to capture logs
    log_capture = StringIO()
    test_handler = logging.StreamHandler(log_capture)
    test_formatter = logging.Formatter('%(levelname)s - %(message)s')
    test_handler.setFormatter(test_formatter)
    
    # Get a clean logger for testing
    test_logger = logging.getLogger("test_shapxplain")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(test_handler)
    test_logger.propagate = False
    
    # Log messages at different levels
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    
    # Get the log output
    log_output = log_capture.getvalue()
    
    # Verify all messages are present
    assert "DEBUG - Debug message" in log_output
    assert "INFO - Info message" in log_output
    assert "WARNING - Warning message" in log_output
    assert "ERROR - Error message" in log_output
    
    # Clean up
    test_logger.handlers.clear()