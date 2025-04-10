"""Tests for the prompt_function decorator."""

import inspect
from typing import Optional

from colloquy_chatbot import prompt_function


def test_prompt_function_basic():
    """Test that prompt_function correctly wraps a function."""
    
    @prompt_function(description="Test function")
    def add(a=1, b=2):
        """Add two numbers."""
        return a + b
    
    # Test function still works
    assert add(3, 4) == 7
    
    # Test metadata is attached
    assert hasattr(add, "__llm_function__")
    assert add.__llm_function__ is True
    
    # Test function name is preserved
    assert add.__name__ == "add"
    
    # Test description is taken from decorator
    assert add.__llm_metadata__["description"] == "Test function"
    
    # Test parameters are inferred
    params = add.__llm_metadata__["parameters"]
    assert "a" in params
    assert "b" in params
    assert params["a"]["type"] == "integer"
    assert params["b"]["type"] == "integer"


def test_prompt_function_with_type_hints():
    """Test that type hints are correctly inferred."""
    
    @prompt_function()
    def process_data(name: str, count: int, active: bool = True, data: Optional[dict] = None):
        """Process some data."""
        return name
    
    params = process_data.__llm_metadata__["parameters"]
    
    assert params["name"]["type"] == "string"
    assert params["count"]["type"] == "integer"
    assert params["active"]["type"] == "boolean"
    assert params["active"]["default"] is True
    assert params["data"]["type"] == "object"


def test_prompt_function_with_parameter_descriptions():
    """Test that parameter descriptions are correctly set."""
    
    @prompt_function(parameter_descriptions={
        "name": "The user's name",
        "age": "The user's age"
    })
    def greet(name="User", age=30):
        return f"Hello {name}, you are {age} years old"
    
    params = greet.__llm_metadata__["parameters"]
    
    assert params["name"]["description"] == "The user's name"
    assert params["age"]["description"] == "The user's age"


def test_prompt_function_docstring():
    """Test that docstring is used for description when not provided."""
    
    @prompt_function()
    def calculate_area(length=1, width=1):
        """Calculate the area of a rectangle."""
        return length * width
    
    assert calculate_area.__llm_metadata__["description"] == "Calculate the area of a rectangle."