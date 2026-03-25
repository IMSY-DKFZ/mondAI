# Tests

## Writing tests

In addition to the `Tests` section in the README, here are some helpful tips for writing tests.

There are `phantom` and `brain` as well as some other fixtures (defined in `conftest.py`) which can be used for testing purposes.

There are three different types of tests:
1. general framework tests: These check that the generic implementations of the base classes, the util functions and the framework in general work correctly. They are located in `test_*.py` files under `tests/`
2. metric specific tests: These perform tests that are specific to individual metrics and are located under `tests/metric_specific_tests/`
3. regression tests: These compare that the implementations are stable, e.g. don't change when dependencies are updated. They are generated in `test_regression.py` and reference scores are stored under `tests/test_regression/`

## Running tests

Run tests with `pytest` and compute the coverage with

> pytest --cov=mondAI --cov-report=term-missing

In addition to tests checking the correctness of the code, there are also regression tests defined in `test_regression.py` which ensure that the computed metrics scores don't change, e.g. when updating dependencies. These scores are computed on the first run of the test (which then fails and creates a reference file under `test_regression`) and then used for consecutive runs for testing. Use `pytest --force-regen` or `pytest --regen-all` to update a single or all test scores. The test scores under `test_regression` need to be committed.
