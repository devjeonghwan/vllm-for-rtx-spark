"""Custom exceptions for handler resolution and loading."""


class HandlerResolutionError(Exception):
    """Base exception for handler resolution errors."""

    pass


class InvalidHandlerSpecError(HandlerResolutionError):
    """Raised when a handler specification is invalid or malformed."""

    def __init__(self, spec: str, reason: str):
        self.spec = spec
        self.reason = reason
        message = f"Invalid handler specification '{spec}': {reason}"
        super().__init__(message)


class HandlerNotFoundError(HandlerResolutionError):
    """Raised when a specified handler cannot be found."""

    def __init__(self, spec: str, source: str):
        self.spec = spec
        self.source = source
        message = f"Handler '{spec}' not found in {source}"
        super().__init__(message)


class HandlerNotCallableError(HandlerResolutionError):
    """Raised when a found handler is not callable."""

    def __init__(self, spec: str, handler_type: str):
        self.spec = spec
        self.handler_type = handler_type
        message = f"Handler '{spec}' is not callable (found {handler_type})"
        super().__init__(message)


class ModuleLoadError(HandlerResolutionError):
    """Raised when a module cannot be loaded."""

    def __init__(self, module_path: str, reason: str):
        self.module_path = module_path
        self.reason = reason
        message = f"Failed to load module '{module_path}': {reason}"
        super().__init__(message)


class HandlerFileNotFoundError(HandlerResolutionError):
    """Raised when a specified file cannot be found."""

    def __init__(self, file_path: str, search_paths: list[str]):
        self.file_path = file_path
        self.search_paths = search_paths
        message = f"File '{file_path}' not found in search paths: {search_paths}"
        super().__init__(message)


# Middleware exceptions
class MiddlewareError(Exception):
    """Base exception for middleware-related errors."""

    pass


class MiddlewareRegistrationError(MiddlewareError):
    """Exception raised when middleware registration fails."""

    pass


class MiddlewareNotFoundError(MiddlewareError):
    """Exception raised when requested middleware is not found."""

    pass


class MiddlewareConfigurationError(MiddlewareError):
    """Exception raised when middleware configuration is invalid."""

    pass


class FormatterRegistrationError(MiddlewareError):
    """Exception raised when formatter registration fails."""

    pass
