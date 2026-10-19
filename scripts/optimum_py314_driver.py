"""Run optimum-cli exports on Python 3.14.

CPython 3.14 made ``functools.partial`` a descriptor (it now has ``__get__``),
so optimum's ``NormalizedConfig.with_args(...)`` class attributes bind ``self``
when accessed through an instance, producing::

    TypeError: NormalizedConfig.__init__() got multiple values for argument 'allow_new'

Until optimum ships a fix, this driver swaps ``with_args`` for a version that
returns a non-descriptor callable (pre-3.14 behavior) BEFORE optimum's config
classes are defined, then hands off to the real CLI::

    python scripts/optimum_py314_driver.py export openvino --model ... --task ...

Harmless on Python <= 3.13. Drop when optimum fixes upstream.
"""
import sys


def _install_partial_shim() -> None:
    import optimum.utils.normalized_config as nc

    class _NoBind:
        """partial-like callable that is NOT a descriptor (pre-3.14 behavior)."""

        def __init__(self, func, keywords):
            self.func, self.keywords = func, keywords

        def __call__(self, *args, **kwargs):
            merged = {**self.keywords, **kwargs}
            return self.func(*args, **merged)

        def __repr__(self):
            return f"_NoBind({self.func!r}, {self.keywords!r})"

    def with_args(cls, allow_new: bool = False, **kwargs):
        return _NoBind(cls, {"allow_new": allow_new, **kwargs})

    nc.NormalizedConfig.with_args = classmethod(with_args)


_install_partial_shim()

from optimum.commands.optimum_cli import main  # noqa: E402  (must import AFTER the shim)

if __name__ == "__main__":
    sys.exit(main())
