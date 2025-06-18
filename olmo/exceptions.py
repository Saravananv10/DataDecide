class OLMoError(Exception):
    """Base class for OLMo exceptions."""

    pass


class OLMoConfigurationError(OLMoError):
    """Raised when there is an error in the OLMo configuration."""

    pass


class OLMoCliError(OLMoError):
    """Raised when there is an error in the OLMo CLI."""

    pass


class OLMoEnvironmentError(OLMoError):
    """Raised when there is an error in the OLMo environment."""

    pass


class OLMoNetworkError(OLMoError):
    """Raised when there is an error in the OLMo network."""

    pass


class OLMoCheckpointError(OLMoError):
    """Raised when there is an error with an OLMo checkpoint."""

    pass
