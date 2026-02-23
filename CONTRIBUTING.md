# Contributing to mondAI

The mondAI library is an open project. We invite you to become a part of it and would love to receive your contributions. There are multiple options for contributing.

## 🐛 Bug Reports

When you've spotted something which doesn't quite work or is completely messed up, please submit a bug report. We can only fix the bugs we know about. A detailed description helps us to reproduce the bug and fix it as soon as possible.

## 🚀 Feature Request

If you have an idea how we could make mondAI even more awesome, feel free to share it by creating an issue. Maybe we or someone from the community finds time to implement it.

## 💻 Contributing Code

Of course also code contributions are welcome. To avoid any frustration on the developer and maintainer side, please stick to the following guidlines.

1. Create an issue first (if it doesn't exist yet) and assign yourself so that no one else is working on the same issue
2. You might have to fork this repo and clone it to be able to work on it.
3. Implement your changes according to coding best practices. Please also write tests for your code and make sure the pipelines pass successfully.
4. If it is your first time contributing to this repo, add your name to the corresponding section in the `CONTRIBUTING.md` file. By doing this, you become one of the authors according to the license.
5. Create a pull request where you describe your changes an reference the issue(s) that this PR solves.
6. Wait until we come back to you. We will do a code review and if we are happy with the contribution, we will merge it. Otherwise we will give you feedback on how to improve your contribution.

Please reach out at any time in case you have questions or need support.

## Reimplementing metrics

Most of the implemented metrics will not be invented newly and also their implementations will be based on already published code. While there is no copyright on metric formulas, the code of specific implementations is protected and maybe licensed. Please stick to these rules:
* Only use code which is compatible to the Apache 2.0 license (e.g. no unlicensed or coply-left licensed code such as GPL)
* Credit and reference the original definition of the metric in the class documentation like this:
```
This implementation follows the definition of XXX as described in:
Author et al., "XXX Metric", Journal, Year
```
* Give credit to the original code authors by:
    * mentioning that the reimplementation is [inspired by/adapted from/taken from] a specific commit from a repository. Please also mention the corresponding license and any modification you did to the implementation for transparency.
    * also add this information to the `THIRD_PARTY_NOTICES.md` file for the corresponding metric

## List of Contributers

* Tom Rix
