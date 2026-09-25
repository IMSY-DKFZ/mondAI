# Logger

To access the logger, you first need to obtain the singelton logger instance by calling `get_logger`:
```python
from mondAI.logging import get_logger

logger = get_logger()
```

This is a `logging.Logger` object which supports `debug`, `info`, `warning`, `error` and `critical` messages by calling the corresponding methods:
```python
logger.debug(f"Image has shape {image.shape}")
logger.warning("The input images might be too small to be processed correctly")
```

As errors often result in incorrect metric scores or even crash their calculation they should rather be `raised` instead of just being logged. Like this the user gets direct feedback that something went wrong and needs to be changed.
