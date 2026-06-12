# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import inspect
import os
import pkgutil
import sys

import mondAI.metrics as metrics_pkg
from mondAI.metrics.base import Metric

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mondAI"
copyright = "2026, Division of Intelligent Systems (IMSY), German Cancer Research Center (DKFZ)"
author = "Division of Intelligent Systems (IMSY), German Cancer Research Center (DKFZ)"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",  # generate documentation using doc strings
    "sphinx.ext.autosummary",  # used to generate overview tables
    "sphinx.ext.intersphinx",  # link between different python packages
    "sphinx.ext.viewcode",  # links to code snippets
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autosummary_generate_overwrite = True

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_sidebars = {}
announcement = (
    "This is a community-supported library. If you'd like to contribute, "
    "<a href='https://git.dkfz.de/imsy/ispai/mondai' target='_blank'>check out our GitLab repository</a>. "
    "Your contributions are welcome!"
)
html_theme_options = {
    # "description": "Correct, reproducible and tested metric implementations for image quality",
    "announcement": announcement,
    "secondary_sidebar_items": {
        "**/*": [
            "page-toc",
            "sourcelink",
        ],
        "index": [],
    },
}

# -- Generating Metric List --


def discover_metrics():
    metric_classes = []

    for _, module_name, _ in pkgutil.walk_packages(
        metrics_pkg.__path__,
        metrics_pkg.__name__ + ".",
    ):
        module = importlib.import_module(module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if inspect.isabstract(obj) or obj.__name__ in ["DISTS", "LPIPS", "NIQE"]:
                continue
            print(obj)
            if issubclass(obj, Metric) and obj is not Metric:
                metric_classes.append(obj())

    # remove metric_classes with duplicate names, keeping only the first occurrence
    seen_names = set()
    unique_metric_classes = []
    for metric in metric_classes:
        if metric.__class__.__name__ not in seen_names:
            unique_metric_classes.append(metric)
            seen_names.add(metric.__class__.__name__)
    return sorted(unique_metric_classes, key=lambda c: c.__class__.__name__)


def generate_metrics_list(app):
    metrics = discover_metrics()

    lines = []
    lines.append("# Metrics List\n")
    lines.append(
        "This is a list of all metrics implemented in the `mondAI` library. For more details on each metric"
        ", click on the name to navigate to the corresponding documentation page.\n"
    )
    lines.append("| Abbreviation | Name | Better | Expected dimensions | Type |")
    lines.append("|--------|------|------------------|---------------|----|")

    for m in metrics:
        cls = f"{m.__class__.__module__}.{m.__class__.__name__}"
        lines.append(
            f"| {m.abbreviation} | {{py:class}}`{m.name} <{cls}>` | "
            f"{'⬆️' if m.higher_is_better else '⬇️'} | {tuple([d.name for d in m.expected_dimensions])} | "
            f" {{py:class}}`{'FR' if 'full_reference' in m.__class__.__module__ else 'NR'}"
            f"<{'.'.join(m.__class__.__module__.split('.')[:-1])}>` |"
        )
    out_path = app.srcdir / "metrics_list.md"
    out_path.write_text("\n".join(lines))


def setup(app):
    app.connect("builder-inited", generate_metrics_list)
