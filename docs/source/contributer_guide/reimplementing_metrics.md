# Implementing metrics

To reduce the burden of implementing a new metric, you can simply copy the `metric_template.py` file and follow the #TODO comments.

## Reimplementing metrics (license issues)

Most of the implemented metrics will not be invented newly and also their implementations will be based on already published code. While there is no copyright on metric formulas, the code of specific implementations is protected and may be licensed. Please stick to these rules:
* Only use code which is compatible to the Apache 2.0 license (e.g. no unlicensed or coply-left licensed code such as GPL)
* Credit and reference the original definition of the metric in the class documentation like this:
```
This implementation follows the definition of XXX as described in:
Author et al., "XXX Metric", Journal, Year
```
* Give credit to the original code authors by:
    * mentioning that the reimplementation is [inspired by/adapted from/taken from] a specific commit from a repository. Please also mention the corresponding license and any modification you did to the implementation for transparency.
    * also add this information to the `THIRD_PARTY_NOTICES.md` file for the corresponding metric
