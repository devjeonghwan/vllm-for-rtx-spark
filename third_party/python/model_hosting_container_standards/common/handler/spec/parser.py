"""Handler specification parser and validator.

Parses and validates handler specifications in these formats:

Function specifications:
- "model.py:predict_fn" → file-based function loading
- "/opt/ml/model.py:predict_fn" → absolute path file-based function loading
- "handler.py:MyClass.process" → file-based class method loading
- "mypackage:handler_fn" → module-based function loading
- "model:predict" → customer script function loading

Router specifications:
- "/health" → router path for health endpoint
- "/v1/chat/completions" → router path for API endpoint
"""

import re
from typing import Optional

from model_hosting_container_standards.exceptions import InvalidHandlerSpecError


class HandlerSpec:
    """Represents a handler specification that can be either a function spec or a router path.

    Examples:
    - HandlerSpec("model.py:predict_fn") → file-based function
    - HandlerSpec("/opt/ml/model.py:predict_fn") → absolute path file-based function
    - HandlerSpec("mypackage:handler") → module-based function
    - HandlerSpec("/health") → router path
    """

    # Regex pattern for valid function specifications
    FUNCTION_SPEC_PATTERN = r"^([^\s:]+):([^\s:]+)$"

    def __init__(self, spec: str):
        """Initialize with a specification string.

        Args:
            spec: The specification string (e.g., "myfile.py:function" or "/health")
        """
        self.spec = spec.strip()

    @property
    def is_router_path(self) -> bool:
        """Check if the spec is a router path (starts with "/" and has no ":")."""
        return self.spec.startswith("/") and ":" not in self.spec

    @property
    def is_function(self) -> bool:
        """Check if the spec is a function specification (contains ":")."""
        return ":" in self.spec

    @property
    def is_callable(self) -> bool:
        """Alias for is_function - check if the spec represents a callable."""
        return self.is_function

    @property
    def is_module_function(self) -> bool:
        """Check if the spec is a module function (contains ":" but no ".py")."""
        return self.is_function and ".py:" not in self.spec

    @property
    def is_file_function(self) -> bool:
        """Check if the spec is a file-based function (contains ".py:")."""
        return self.is_function and ".py:" in self.spec

    @property
    def is_class_method(self) -> bool:
        """Check if the spec appears to be a class method (contains "." after ":")."""
        if not self.is_function:
            return False
        _, func_part = self.spec.split(":", 1)
        return "." in func_part

    @property
    def module_name(self) -> Optional[str]:
        """Extract module name from module function spec."""
        if self.is_module_function:
            return self.spec.split(":", 1)[0]
        return None

    @property
    def file_path(self) -> Optional[str]:
        """Extract file path from file function spec."""
        if self.is_file_function:
            return self.spec.split(":", 1)[0]
        return None

    @property
    def function_name(self) -> Optional[str]:
        """Extract function name from function spec."""
        if self.is_function:
            return self.spec.split(":", 1)[1]
        return None

    @property
    def class_name(self) -> Optional[str]:
        """Extract class name from class method spec."""
        if self.is_class_method:
            func_part = self.function_name
            if func_part and "." in func_part:
                return func_part.split(".", 1)[0]
        return None

    @property
    def method_name(self) -> Optional[str]:
        """Extract method name from class method spec."""
        if self.is_class_method:
            func_part = self.function_name
            if func_part and "." in func_part:
                return func_part.split(".", 1)[1]
        return None

    @property
    def router_path(self) -> Optional[str]:
        """Get the router path if this is a router spec."""
        return self.spec if self.is_router_path else None

    def __str__(self) -> str:
        """String representation of the spec."""
        return self.spec

    def __repr__(self) -> str:
        """Detailed representation of the spec."""
        spec_type = "router_path" if self.is_router_path else "function"
        return f"HandlerSpec('{self.spec}', type={spec_type})"

    def is_valid_function_spec(self) -> bool:
        """Check if this is a valid function specification."""
        # Try to validate as function spec
        try:
            self.validate_function_spec()
            return True
        except InvalidHandlerSpecError:
            return False

    def validate_function_spec(self) -> tuple[str, str]:
        """Validate and parse function specification.

        Returns:
            Tuple of (module_or_file_path, function_name)

        Raises:
            InvalidHandlerSpecError: If the specification is invalid
        """
        if self.is_router_path:
            raise InvalidHandlerSpecError(
                self.spec,
                f"Cannot validate router path '{self.spec}' as function specification",
            )

        # Use the class constant pattern for validation
        match = re.match(self.FUNCTION_SPEC_PATTERN, self.spec)

        if not match:
            raise InvalidHandlerSpecError(
                self.spec,
                f"Function spec must match pattern: {self.FUNCTION_SPEC_PATTERN} "
                f"(format: 'module:function')",
            )

        return match.group(1), match.group(2)


def parse_handler_spec(spec_string: Optional[str]) -> Optional[HandlerSpec]:
    """Parse a handler specification string into a HandlerSpec object.

    Args:
        spec_string: The specification string or None

    Returns:
        HandlerSpec object if spec_string is provided, None otherwise
    """
    if not spec_string:
        return None
    return HandlerSpec(spec_string)
