"""Expected operational failures.

These are raised for missing indexes, bad configuration, and provider
errors. Callers turn them into HTTP responses or CLI exit codes. They are
not used for programmer errors.
"""


class CiteGroundError(Exception):
    """Base class for errors the operator can act on."""


class IndexNotFoundError(CiteGroundError):
    """Raised when the on-disk index has not been built."""


class ProviderError(CiteGroundError):
    """Raised when an embedding or chat provider cannot complete a call."""


class ConfigError(CiteGroundError):
    """Raised when a setting or experiment spec is inconsistent."""
