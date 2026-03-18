# Limitations

Currently, the package does not support
* masked image metrics
* images with `NaN` values
* temporal images (such as videos or longitudinal images)

At the moment the user is responsible for **normalizing** the images to the correct value ranges which are expected by metrics. Metrics inform the user if values are outside of valid ranges by throwing errors, and sometimes even give warnings if they detect that for example all values are between [0, 1] but should be between [0, 255].

Furthermore, it's the responsibility of the user to aggregate returned metric scores, e.g. over channel dimensions or over samples of a batch. There is so far no implemented reduction function. But simply calling `torch.mean()` or `torch.sum()` or whatever with the corresponding dimension will do the job.
