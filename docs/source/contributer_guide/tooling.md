# Tooling

This project uses some helpful tools which are listed here:

## Pre-commit hooks

To keep the repository and its code clean, pre-commit hooks are provided. You can install them by running
> pre-commit install

in the repositories root. The hooks will then trigger each time when you try to commit something. If they fail, they give you an error message and sometimes even automatically fix the issues so that you can directly retry to commit the code.

To trigger the hooks manually, run
> pre-commit run --all-files

## Ruff linter and formatter

To keep the code valid and maintain a consistent code style and format ruff is used. Besides the opportunity to run it as an IDE plugin, ruff will be called as a pre-commit hook.

## Pytest

For testing purposes pytest is used. It also computes the test coverage which will be shown for the `dev` branch in the repo or for each commit in the corresponding CI/CD jobs. To read more about testing, see [tests](tests.md).

## Documentation

This documentation is created with Sphinx and the PyDataTheme. To read more about documentation, see [documentation](documentation.md).

## CI/CD pipeline

Linting, testing, documentation build and deployment are run automatically upon commit and push to the repo.
