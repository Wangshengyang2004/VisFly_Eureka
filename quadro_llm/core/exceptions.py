"""
Custom exception classes and error handling utilities for VisFly-Eureka

This module defines a hierarchy of custom exceptions and provides utilities
for consistent error handling across the codebase.
"""

import logging
from typing import Optional, Dict, Any


class VisFlyEurekaError(Exception):
    """Base exception class for all VisFly-Eureka errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(VisFlyEurekaError):
    """Raised when there's an error in configuration"""
    pass


class EnvironmentError(VisFlyEurekaError):
    """Raised when there's an error creating or using environments"""
    pass


class RewardInjectionError(VisFlyEurekaError):
    """Raised when reward function injection fails"""
    pass


class LLMError(VisFlyEurekaError):
    """Raised when LLM operations fail"""
    pass


class APIConnectionError(LLMError):
    """Raised when API connection fails"""
    pass


class TrainingError(VisFlyEurekaError):
    """Raised when training operations fail"""
    pass


class EvaluationError(VisFlyEurekaError):
    """Raised when evaluation operations fail"""
    pass


class GPUResourceError(VisFlyEurekaError):
    """Raised when GPU resource operations fail"""
    pass


class CUDAOutOfMemoryError(GPUResourceError):
    """Raised when CUDA runs out of memory"""
    pass


class SubprocessError(VisFlyEurekaError):
    """Raised when subprocess operations fail"""
    pass


class ValidationError(VisFlyEurekaError):
    """Raised when input validation fails"""
    pass


# Error handling utilities

def handle_and_log_error(
    logger: logging.Logger,
    error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
) -> None:
    """
    Standardized error handling and logging
    
    Args:
        logger: Logger instance
        error: The caught exception
        operation: Description of what operation was being performed
        context: Additional context information
        reraise: Whether to reraise the exception after logging
    """
    context = context or {}
    
    if isinstance(error, VisFlyEurekaError):
        # Our custom exceptions already have good structure
        logger.error(
            "Operation '%s' failed: %s", 
            operation, 
            error,
            extra={"error_details": error.details, "context": context}
        )
    else:
        # Handle standard exceptions
        logger.exception(
            "Operation '%s' failed with %s: %s",
            operation,
            type(error).__name__,
            str(error),
            extra={"context": context}
        )
    
    if reraise:
        raise


def convert_to_domain_error(
    error: Exception,
    operation: str,
    error_type: type = VisFlyEurekaError,
    additional_details: Optional[Dict[str, Any]] = None
) -> VisFlyEurekaError:
    """
    Convert a generic exception to a domain-specific exception
    
    Args:
        error: The original exception
        operation: Description of the operation that failed
        error_type: The domain exception type to convert to
        additional_details: Additional context to include
    
    Returns:
        Domain-specific exception
    """
    details = additional_details or {}
    details["original_error_type"] = type(error).__name__
    details["operation"] = operation
    
    message = f"Operation '{operation}' failed: {str(error)}"
    
    return error_type(
        message=message,
        details=details,
        cause=error
    )


def safe_execute(
    operation_func,
    operation_name: str,
    logger: logging.Logger,
    error_type: type = VisFlyEurekaError,
    context: Optional[Dict[str, Any]] = None,
    default_return=None
):
    """
    Safely execute an operation with standardized error handling
    
    Args:
        operation_func: Function to execute
        operation_name: Name of the operation for logging
        logger: Logger instance
        error_type: Type of exception to raise on error
        context: Additional context for error reporting
        default_return: Value to return on error (if None, reraises)
    
    Returns:
        Result of operation_func or default_return on error
    """
    try:
        return operation_func()
    except Exception as e:
        domain_error = convert_to_domain_error(
            error=e,
            operation=operation_name,
            error_type=error_type,
            additional_details=context
        )
        
        handle_and_log_error(
            logger=logger,
            error=domain_error,
            operation=operation_name,
            context=context,
            reraise=(default_return is None)
        )
        
        return default_return


# Context managers for error handling

class ErrorContext:
    """Context manager for standardized error handling"""
    
    def __init__(
        self,
        operation: str,
        logger: logging.Logger,
        error_type: type = VisFlyEurekaError,
        context: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.logger = logger
        self.error_type = error_type
        self.context = context or {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            domain_error = convert_to_domain_error(
                error=exc_val,
                operation=self.operation,
                error_type=self.error_type,
                additional_details=self.context
            )
            
            handle_and_log_error(
                logger=self.logger,
                error=domain_error,
                operation=self.operation,
                context=self.context,
                reraise=True
            )
        return False  # Don't suppress exceptions


# Common error patterns

def validate_not_none(value, name: str):
    """Validate that a value is not None"""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def validate_file_exists(file_path: str):
    """Validate that a file exists"""
    import os
    if not os.path.exists(file_path):
        raise ValidationError(
            f"File not found: {file_path}",
            details={"file_path": file_path}
        )


def validate_positive_number(value: float, name: str):
    """Validate that a number is positive"""
    if value <= 0:
        raise ValidationError(
            f"{name} must be positive, got {value}",
            details={"value": value, "parameter": name}
        )