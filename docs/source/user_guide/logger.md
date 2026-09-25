# Logger

By default, mondAI logs many messages at the INFO level during metric computation to the command line keep users informed. But users might be interested in more detail, including the DEBUG logging level, or prefer to only receive WARNING messages. The logging level can be changed package wide like this:

```python
from mondAI.settings import settings
settings.logging_level = "DEBUG"
# or
settings.logging_level = "WARNING"
```
