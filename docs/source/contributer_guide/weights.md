# Storing Weights

Some metrics have parameters, weights or other statistics they use during computation which were learned or acquired beforehand. If the size of these files is small (e.g. < 1 MB) they can be stored within the package in the `metrics/weights/` folder. Then you can load the file with:
```python
str(PurePosixPath(Path.cwd() / f"src/mondAI/metrics//weights/yourfile.name"))
```

In case the file(s) are too large, they should be downloaded upon first usage from a given url and stored in the torch hub `mondAI` folder:

```python
from torch.hub import get_dir, load_state_dict_from_url

# download model weights if not already downloaded and store in cache directory, and load weights
model_weights_path = os.path.join(get_dir(), "mondAI")
try:
    logger.info(
        f"Loading model weights from {model_weights_path}. "
        f"If not present, downloading from {self.model_weights_url}..."
    )
    model_state_dict = load_state_dict_from_url(
        self.model_weights_url,
        model_dir=model_weights_path,
        map_location=lambda storage, loc: storage,
        progress=True,
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to download and load model weights from {self.model_weights_url}. "
        f"Please check the URL and your internet connection. Error: {e}"
    ) from e
```
