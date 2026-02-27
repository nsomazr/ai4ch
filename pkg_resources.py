"""
Minimal stub implementation of the pkg_resources API used by django-countries.

On some Debian/Ubuntu systems with Python 3.12, ``pkg_resources`` is provided
by a separate system package that may not be available inside application
virtualenvs. This lightweight shim is enough for django-countries, which only
relies on ``iter_entry_points`` to discover optional extensions.

By returning an empty iterator, we simply disable extension discovery without
breaking core functionality.
"""

from typing import Iterable, Any


def iter_entry_points(group: str, name: str | None = None) -> Iterable[Any]:
    """
    Return an empty iterator for entry points.

    django-countries calls:

        pkg_resources.iter_entry_points(\"django_countries.Country\")

    to look for custom Country implementations. Returning no entry points is
    safe and just means no extensions are loaded.
    """
    return ()

