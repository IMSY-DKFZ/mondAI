# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0
"""Test the mondAI package's __version__ attribute."""


def test_version() -> None:
    from mondAI import __version__

    assert isinstance(__version__, str)
