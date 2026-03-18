# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

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
