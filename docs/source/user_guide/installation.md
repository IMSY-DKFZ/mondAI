# Installation

To install `mondAI` from PyPI run (not working yet)
> pip install mondAI

To install it from gitlab, run
> pip install git+https://gitlab.com/dkfz_imsy/mondai.git

To install the package directly from code, clone the repository and run
> git clone https://gitlab.com/dkfz_imsy/mondai.git
> cd mondai
> pip install .

Add `comparison` if you want to make use of the `compare_implementations=True` feature. This will install additional dependencies.
> pip install mondAI[comparison]
or
> pip install .[comparison]

## Development Version
To install the editable development version, clone the repository and run
> git clone https://gitlab.com/dkfz_imsy/mondai.git
> cd mondai
> pip install -e .[dev,comparison]
