from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Iterable, List, Optional
import torch.nn as nn
import importlib
from pathlib import Path


class NormLinearIterator(ABC):
    """iterate over norm and its subsequent linear layers"""
    
    _registered_iterators: List["NormLinearIterator"] = []
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[nn.Module, str, Iterable[nn.Linear]]]:
        """(parent_module, norm_layer_name, [linear_layers])"""
        pass
    
    @classmethod
    @abstractmethod
    def supports_model(cls, model: nn.Module) -> bool:
        """check if the model is supported"""
        pass
    
    @classmethod
    def register_iterator(cls, iter_cls) -> "NormLinearIterator":
        """register an iterator class"""
        cls._registered_iterators.append(iter_cls)
        return iter_cls
    
    @classmethod
    def from_model(cls, model: nn.Module) -> "NormLinearIterator":
        for iterator_cls in cls._registered_iterators:
            if iterator_cls.supports_model(model):
                return iterator_cls(model)
        
        raise ValueError(
            f"No suitable NormLinearIterator found for model type {type(model)}. "
            "Consider implementing and registering a custom iterator."
        )


from typing import Dict, Type, Callable, Any, Union
import torch
import torch.nn as nn

class AutoOperationMeta(type):
    def __getattr__(cls, name):
        if name.startswith('_'):
            raise AttributeError(name)
        
        def method(module: nn.Module, *args, **kwargs):
            return cls.apply_operation(name, module, *args, **kwargs)
        
        return method

class AutoOperation(metaclass=AutoOperationMeta):
    """
    A class that supports registering and applying operations to different module types.
    Operations can be registered externally and applied to modules dynamically.
    """
    
    # Nested dictionary to store operations:
    # {operation_name: {module_type: operation_func}}
    _operations: Dict[str, Dict[Type[nn.Module], Callable]] = {}
    
    @classmethod
    def register_operation(cls, operation_name: str, module_type: Type[nn.Module]):
        """
        Decorator to register an operation for a specific module type.
        
        Args:
            operation_name: Name of the operation (e.g., 'rotate_input')
            module_type: The module type this operation applies to
        """
        def decorator(func: Callable):
            if operation_name not in cls._operations:
                cls._operations[operation_name] = {}
            cls._operations[operation_name][module_type] = func
            return func
        return decorator
    
    @classmethod
    def apply_operation(cls, operation_name: str, module: nn.Module, *args, **kwargs) -> Any:
        """
        Apply a registered operation to a module.
        
        Args:
            operation_name: Name of the operation to apply
            module: The module to apply the operation to
            *args, **kwargs: Additional arguments to pass to the operation
            
        Returns:
            The result of the operation (if any)
            
        Raises:
            ValueError: If the operation is not registered for the module type
        """
        if operation_name not in cls._operations:
            raise ValueError(f"Operation '{operation_name}' not registered")
            
        module_type = type(module)
        for base in module_type.__mro__:
            if base in cls._operations[operation_name]:
                return cls._operations[operation_name][base](module, *args, **kwargs)
        
        raise ValueError(f"Operation '{operation_name}' not registered for module type {module_type}")
    
    @classmethod
    def has_operation(cls, operation_name: str, module_type: Type[nn.Module]) -> bool:
        """
        Check if an operation is registered for a module type.
        """
        if operation_name not in cls._operations:
            return False
        return any(base in cls._operations[operation_name] for base in module_type.__mro__)
    
    # Convenience methods (dynamically generated based on registered operations)
    def __getattr__(cls, name):
        if name.startswith('_'):
            raise AttributeError(name)
        
        def method(module: nn.Module, *args, **kwargs):
            return cls.apply_operation(name, module, *args, **kwargs)
        
        return method

@AutoOperation.register_operation("rotate_input", nn.Linear)
def op_rotate_linear_input(
    linear: torch.nn.Linear,
    R: torch.Tensor):
    """
    Rotate the input of linear layers by a rotation matrix.
    i.e. xW + b -> (xR)W + b ==> x(RW) + b
    This is done by multiplying the weight matrix by the rotation matrix.
    The rotation matrix should be orthogonal.
    """
    R_dim = R.shape[0]
    in_dim = linear.in_features
    repeat_times = in_dim // R_dim
    assert in_dim % R_dim == 0, "input dim should be multiple of rotation matrix dim"
    # refer to patch merger of ViT of Qwen2VL
    R = torch.block_diag(*([R] * repeat_times))  # sometimes we calculate (x1R, x2R, x3R) W + b, which is equivalent to (x1, x2, x3) diag(R, R, R) W + b
    dtype = linear.weight.dtype
    R_device = R.device
    w_device = linear.weight.device
    W_ = linear.weight.data.to(device=R_device, dtype=torch.float64)
    # note that the W_ in linear is transpose of W
    linear.weight.data = (W_ @ (R.T.to(torch.float64))).to(device=w_device, dtype=dtype)
        

@AutoOperation.register_operation("rotate_output", nn.Linear)
def op_rotate_linear_output(
    linear: nn.Linear,
    R: torch.Tensor):
    """
    Rotate the output of linear layers by a rotation matrix.
    i.e. o = xW + b -> o = (xW + b)R ==> o = x(WR) + bR
    This is done by multiplying the weight matrix by the rotation matrix.
    The rotation matrix should be orthogonal.
    """
    assert R.shape[0] == R.shape[1], "R should be a square matrix"
    assert R.shape[0] == linear.weight.shape[0], "R should be same size as output dim of linear layer"
    dtype = linear.weight.dtype
    R_device = R.device
    w_device = linear.weight.device
    W_ = linear.weight.data.to(device=R_device, dtype=torch.float64)
    # note that the W_ in linear is transpose of W
    linear.weight.data = (R.T.to(torch.float64) @ W_).to(device=w_device, dtype=dtype)
    # rotate the bias
    if linear.bias is not None:
        bias = linear.bias.data.to(device=R_device, dtype=torch.float64)
        linear.bias.data = (bias @ R.to(torch.float64)).to(device=linear.bias.device, 
                                                            dtype=linear.bias.dtype)
    
@AutoOperation.register_operation("rotate_output", nn.Embedding)
def op_rotate_embedding(
    embedding: torch.nn.Embedding,
    R: torch.Tensor):
    """
    Rotate each embedding vector by a rotation matrix R.
    """
    dtype = embedding.weight.dtype
    R_device = R.device
    w_device = embedding.weight.device
    W_ = embedding.weight.data.to(device=R_device, dtype=torch.float64)
    # note that the W_ in linear is transpose of W
    embedding.weight.data = (W_ @ (R.to(torch.float64))).to(device=w_device, dtype=dtype)

        
   
"""
# denote centering the vector x as C(x) = x - mu
# we have C(x) = x - mu = x - mu 1 where 1 is the vector of ones
#  = x - 1/d sum(x) 1
# we have sum(x) = x_1 + x_2 + ... + x_n = 1^T x
# so we have C(x) = x - 1/d (1^T x) 1 = x - 1/d 1 (1^T x) = x - 1/d 1 1^T x
# that is, we can write C(x) = (I - 1/d 1 1^T) x
# denote the matrix I - 1/d 1 1^T as C
# we have C(x) = C x
# here all the vectors are column vectors
# it is easy to see that C is a symmetric matrix
# so for a row vector x we have C(x) = x C^T = x C
"""
@AutoOperation.register_operation("center_output", nn.Linear)
def op_center_linear_output(linear: torch.nn.Linear):
    """
    Center the output of linear layers
    i.e. xW + b -> (xW + b) C = xW C + bC
    that is we need to center the weight matrix by row and the bias
    """
    dtype = linear.weight.dtype
    W_ = linear.weight.data.to(dtype=torch.float64)
    # note that the W_ in linear is transpose of W
    # center echo columns of W equivalent to centering the rows of W_
    W_mean = W_.mean(dim=0, keepdim=True)
    W_centered = W_ - W_mean
    linear.weight.data = W_centered.to(dtype=dtype)
    if linear.bias is not None:
        bias = linear.bias.data.to(dtype=torch.float64)
        bias_mean = bias.mean()
        bias_centered = bias - bias_mean
        linear.bias.data = bias_centered.to(dtype=dtype)   


from typing import Callable, Dict, List, Any, Type
from functools import wraps


class RotateOperationRegistry:
    """A singleton registry for managing rotate operations across different modules.

    This registry maintains a mapping from module types to lists of rotate operations.
    Each module type can have multiple rotate operations registered, which will be
    executed in registration order when the rotate interface is called.
    """

    _instance = None
    _registry: Dict[Type, List[Callable[..., Any]]] = {}

    def __new__(cls):
        """Ensures singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, module_type: Type) -> Callable:
        """Decorator to register a rotate operation for a specific module type.

        Args:
            module_type: The module class that this operation applies to.

        Returns:
            A decorator function that will register the target function.
        """

        def decorator(func: Callable[..., Any]) -> Callable:
            """Inner decorator that performs the actual registration.

            Args:
                func: The rotate operation function to be registered.

            Returns:
                The original function with registration side-effect.
            """
            if module_type not in cls._registry:
                cls._registry[module_type] = []
            cls._registry[module_type].append(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        return decorator

    @classmethod
    def get_operations(cls, module_type: Type) -> List[Callable[..., Any]]:
        """Retrieves all registered rotate operations for a module type.

        Args:
            module_type: The module class to look up operations for.

        Returns:
            A list of registered rotate operations for the given module type.
            Returns empty list if no operations are registered.
        """
        return cls._registry.get(module_type, [])

    @classmethod
    def clear(cls) -> None:
        """Clears all registered operations (primarily for testing purposes)."""
        cls._registry.clear()
        
    @classmethod
    def discover_and_register(cls, package_path: str, module_prefix: str) -> None:
        """Scan a directory and import all Python modules for registration.
        
        Args:
            package_path: Filesystem path to the directory containing registration files.
            module_prefix: Python import path prefix (e.g., 'myapp.registrations').
        """
        package_dir = Path(package_path)
        
        for module_file in package_dir.glob('*.py'):
            if module_file.name.startswith('_'):
                continue  # Skip __init__.py and similar files
            
            module_name = f"{module_prefix}.{module_file.stem}"
            try:
                importlib.import_module(module_name)
                print(f"Imported registration module: {module_name}")
            except ImportError as e:
                print(f"Failed to import {module_name}: {str(e)}")

    @classmethod
    def auto_discover(cls, 
                    package_name: str = "registrations",
                    base_package: Optional[str] = None) -> None:
        """Automatically discover and load registration modules.
        
        Args:
            package_name: Subpackage name containing registrations.
            base_package: Root package name (e.g., 'myapp').
                         If None, attempts to detect from caller's package.
        """
        if base_package is None:
            # Automatic base package detection
            import inspect
            frame = inspect.currentframe()
            try:
                caller_module = inspect.getmodule(frame.f_back)
                base_package = caller_module.__package__.split('.')[0]
            finally:
                del frame  # Clean up to avoid reference cycles
        
        full_package = f"{base_package}.{package_name}" if base_package else package_name
        
        try:
            package = importlib.import_module(full_package)
            package_path = Path(package.__file__).parent
            
            print(f"Discovering modules in: {package_path}")
            cls.discover_and_register(str(package_path), full_package)
        except ImportError as e:
            print(f"Registration package not found: {full_package}: {str(e)}")


def rotate_model(module: Any, *args, **kwargs) -> None:
    """Unified interface to execute all registered rotate operations for a module.

    Executes all registered rotate operations in registration order, passing through
    all provided arguments to each operation.

    Args:
        module: The module instance to be rotated.
        *args: Positional arguments to pass to rotate operations.
        **kwargs: Keyword arguments to pass to rotate operations.

    Raises:
        ValueError: If no rotate operations are registered for the module's type.
    """
    module_type = type(module)
    operations = RotateOperationRegistry.get_operations(module_type)

    if not operations:
        raise ValueError(f"No rotate operations registered for module type: {module_type.__name__}")

    for operation in operations:
        operation(module, *args, **kwargs)


    