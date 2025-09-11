# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""
A dictionary-like class for storing key-value pairs where values can be
torch tensors, numpy ndarrays, or other library tensors/ndarrays.
"""

from typing import Any, Dict, Union, Iterator, Tuple, Optional, List
import warnings
import json
import os
from pathlib import Path

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. torch.Tensor support disabled.")

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. numpy.ndarray support disabled.")

try:
    from safetensors import safe_open
    import safetensors.torch as safetensors_torch
    import safetensors.numpy as safetensors_numpy

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    warnings.warn("safetensors not available. safetensors support disabled.")

# Type alias for supported tensor types
TensorType = Any


def _is_valid_tensor(value: Any) -> bool:
    """Check if value is a supported tensor type."""
    if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return True
    if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
        return True
    # Add checks for other tensor libraries here
    return False


class ParamsDict:
    """
    A dictionary-like class that stores key-value pairs where values can be
    torch tensors, numpy ndarrays, or other library tensors/ndarrays.

    Examples:
        >>> params = ParamsDict()
        >>> params['weight'] = torch.tensor([1, 2, 3])
        >>> params['bias'] = np.array([0.1, 0.2, 0.3])
        >>> len(params)
        2
        >>> 'weight' in params
        True
        >>> params.keys()
        dict_keys(['weight', 'bias'])
    """

    def __init__(self, initial_dict: Optional[Dict[str, TensorType]] = None):
        """Initialize a ParamsDict, optionally with initial data."""
        self._data: Dict[str, TensorType] = {}
        if initial_dict is not None:
            self.update(initial_dict)

    def __setitem__(self, key: str, value: TensorType) -> None:
        """
        Set a key-value pair in the dictionary.

        Args:
            key: The key (string) to store the tensor under
            value: The tensor value (torch.Tensor, numpy.ndarray, etc.)

        Raises:
            TypeError: If the value is not a supported tensor type
            TypeError: If the key is not a string
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")

        if not _is_valid_tensor(value):
            raise TypeError(f"Value must be a supported tensor type, got {type(value)}")

        self._data[key] = value

    def __getitem__(self, key: str) -> TensorType:
        """
        Get a tensor value by key.

        Args:
            key: The key to look up

        Returns:
            The tensor value associated with the key

        Raises:
            KeyError: If the key is not found
        """
        return self._data[key]

    def __delitem__(self, key: str) -> None:
        """
        Delete a key-value pair from the dictionary.

        Args:
            key: The key to delete

        Raises:
            KeyError: If the key is not found
        """
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the dictionary.

        Args:
            key: The key to check for

        Returns:
            True if the key exists, False otherwise
        """
        return key in self._data

    def __len__(self) -> int:
        """
        Get the number of key-value pairs in the dictionary.

        Returns:
            The number of items
        """
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the keys in the dictionary.

        Returns:
            An iterator over the keys
        """
        return iter(self._data)

    def keys(self):
        """
        Get a view of the dictionary's keys.

        Returns:
            A dict_keys view of the keys
        """
        return self._data.keys()

    def values(self):
        """
        Get a view of the dictionary's values.

        Returns:
            A dict_values view of the values
        """
        return self._data.values()

    def items(self):
        """
        Get a view of the dictionary's key-value pairs.

        Returns:
            A dict_items view of the key-value pairs
        """
        return self._data.items()

    def get(self, key: str, default=None) -> TensorType:
        """
        Get a tensor value by key, with a default if not found.

        Args:
            key: The key to look up
            default: The default value to return if key is not found

        Returns:
            The tensor value associated with the key, or the default
        """
        return self._data.get(key, default)

    def clear(self) -> None:
        """Remove all key-value pairs from the dictionary."""
        self._data.clear()

    def update(self, other: Union["ParamsDict", Dict[str, TensorType]]) -> None:
        """
        Update the dictionary with key-value pairs from another ParamsDict or dict.

        Args:
            other: Another ParamsDict or dict to update from

        Raises:
            TypeError: If any value in other is not a supported tensor type
        """
        if isinstance(other, ParamsDict):
            self._data.update(other._data)
        else:
            for key, value in other.items():
                self[key] = value  # Uses __setitem__ for validation

    def copy(self) -> "ParamsDict":
        """
        Create a shallow copy of the ParamsDict.

        Returns:
            A new ParamsDict with the same key-value pairs
        """
        new_dict = ParamsDict()
        new_dict._data = self._data.copy()
        return new_dict

    def to_dict(self) -> Dict[str, TensorType]:
        """
        Return a copy of the internal dictionary.

        Returns:
            A copy of the internal dictionary
        """
        return self._data.copy()

    def __repr__(self) -> str:
        """
        Get a string representation of the ParamsDict.

        Returns:
            A string representation
        """
        items = [
            f"'{k}': {type(v).__name__}{list(v.shape) if hasattr(v, 'shape') else ''}"
            for k, v in self._data.items()
        ]
        items_str = ", ".join(items)
        return f"ParamsDict({{{items_str}}})"

    def __str__(self) -> str:
        """
        Get a string representation of the ParamsDict.

        Returns:
            A string representation
        """
        return self.__repr__()

    def save_pytorch(self, file_path: str) -> None:
        """
        Save the parameters to a PyTorch file.

        Args:
            file_path: Path to save the PyTorch file

        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot save PyTorch format.")

        # Convert all tensors to PyTorch tensors if needed
        torch_dict = {}
        for key, value in self._data.items():
            if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                torch_dict[key] = value
            elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                torch_dict[key] = torch.from_numpy(value)
            else:
                raise TypeError(f"Unsupported tensor type for key {key}: {type(value)}")

        torch.save(torch_dict, file_path)

    def save_safetensors(
        self, file_path: str, metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Save the parameters to a safetensors file.

        Args:
            file_path: Path to save the safetensors file
            metadata: Optional metadata to include in the file

        Raises:
            ImportError: If safetensors is not available
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors is not available. Cannot save safetensors format."
            )

        # Convert to appropriate format for safetensors
        if TORCH_AVAILABLE:
            # Convert all tensors to PyTorch tensors
            torch_dict = {}
            for key, value in self._data.items():
                if isinstance(value, torch.Tensor):
                    torch_dict[key] = value
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    torch_dict[key] = torch.from_numpy(value)
                else:
                    raise TypeError(
                        f"Unsupported tensor type for key {key}: {type(value)}"
                    )

            safetensors_torch.save_file(torch_dict, file_path, metadata=metadata)
        elif NUMPY_AVAILABLE:
            # Convert all tensors to numpy arrays
            numpy_dict = {}
            for key, value in self._data.items():
                if isinstance(value, np.ndarray):
                    numpy_dict[key] = value
                elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                    numpy_dict[key] = value.numpy()
                else:
                    raise TypeError(
                        f"Unsupported tensor type for key {key}: {type(value)}"
                    )

            safetensors_numpy.save_file(numpy_dict, file_path, metadata=metadata)
        else:
            raise ImportError(
                "Neither PyTorch nor NumPy is available. Cannot save safetensors format."
            )

    def to(self, device: str) -> "ParamsDict":
        """
        Move all tensors to the specified device (PyTorch only).

        Args:
            device: The device to move tensors to (e.g., 'cpu', 'cuda')

        Returns:
            self for method chaining

        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not available. Cannot move tensors to device."
            )

        for key in self._data:
            if isinstance(self._data[key], torch.Tensor):
                self._data[key] = self._data[key].to(device)

        return self

    def numpy(self) -> "ParamsDict":
        """
        Convert all tensors to numpy arrays (in-place).

        Returns:
            self for method chaining

        Raises:
            ImportError: If NumPy is not available
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is not available. Cannot convert to numpy arrays.")

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. No tensors to convert.")

        for key in self._data:
            if isinstance(self._data[key], torch.Tensor):
                self._data[key] = self._data[key].numpy()

        return self

    def torch(self) -> "ParamsDict":
        """
        Convert all tensors to PyTorch tensors (in-place).

        Returns:
            self for method chaining

        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not available. Cannot convert to PyTorch tensors."
            )

        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is not available. No arrays to convert.")

        for key in self._data:
            if isinstance(self._data[key], np.ndarray):
                self._data[key] = torch.from_numpy(self._data[key])

        return self


def load_torch_model(model_path: str) -> ParamsDict:
    """
    Load a PyTorch model from a file path into a ParamsDict.

    Args:
        model_path: Path to the PyTorch model file (.pt, .pth, etc.)

    Returns:
        ParamsDict containing the model parameters

    Raises:
        ImportError: If PyTorch is not available
        FileNotFoundError: If the model file is not found
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not available. Please install PyTorch to use this function."
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the PyTorch model state dict
    state_dict = torch.load(model_path, map_location="cpu")

    # Create a ParamsDict and populate it with the model parameters
    return ParamsDict(state_dict)


def _load_safetensors_shard(file_path: str, framework: str = "pt") -> Dict[str, Any]:
    """
    Load a single safetensors shard file.

    Args:
        file_path: Path to the safetensors file
        framework: The framework to load tensors for ("pt" for PyTorch, "np" for NumPy)

    Returns:
        Dictionary containing the loaded tensors
    """
    if framework == "pt":
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not available. Cannot load with framework='pt'."
            )

        from safetensors.torch import load_file as load_torch_file

        return load_torch_file(file_path)
    else:
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is not available. Cannot load with framework='np'."
            )

        from safetensors.numpy import load_file as load_numpy_file

        return load_numpy_file(file_path)


def load_safetensors_model(file_path: str, framework: str = "pt") -> ParamsDict:
    """
    Load a safetensors model from a file path into a ParamsDict.
    Supports both single .safetensors files and .safetensors.index.json index files.

    Args:
        file_path: Path to the safetensors model file (.safetensors) or index file (.safetensors.index.json)
        framework: The framework to load tensors for ("pt" for PyTorch, "np" for NumPy)

    Returns:
        ParamsDict containing the model parameters

    Raises:
        ImportError: If safetensors is not available
        FileNotFoundError: If the model file is not found
        ValueError: If framework is invalid or tensor validation fails
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError(
            "safetensors is not available. Please install safetensors to use this function."
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    if framework not in ["pt", "np"]:
        raise ValueError(f"Framework must be 'pt' or 'np', got {framework}")

    # Check if this is an index file
    if file_path.endswith(".index.json") or file_path.endswith(
        ".safetensors.index.json"
    ):
        return _load_safetensors_index(file_path, framework)
    else:
        # Single safetensors file
        try:
            state_dict = _load_safetensors_shard(file_path, framework)
            return ParamsDict(state_dict)
        except Exception as e:
            raise ValueError(f"Error loading safetensors file: {e}")


def _load_safetensors_index(index_file_path: str, framework: str = "pt") -> ParamsDict:
    """
    Load a safetensors model from an index file.

    Args:
        index_file_path: Path to the .safetensors.index.json file
        framework: The framework to load tensors for ("pt" for PyTorch, "np" for NumPy)

    Returns:
        ParamsDict containing the model parameters

    Raises:
        ValueError: If the index file format is invalid
    """
    # Load the index file
    with open(index_file_path, "r") as f:
        index_data = json.load(f)

    # Validate index file structure
    if "weight_map" not in index_data:
        raise ValueError("Invalid index file: missing 'weight_map' field")

    if "metadata" not in index_data:
        raise ValueError("Invalid index file: missing 'metadata' field")

    # Get the directory containing the index file
    index_dir = os.path.dirname(index_file_path)

    # Group tensors by shard file
    shard_files = {}
    for tensor_name, shard_file in index_data["weight_map"].items():
        if shard_file not in shard_files:
            shard_files[shard_file] = []
        shard_files[shard_file].append(tensor_name)

    # Load each shard and extract the required tensors
    params_dict = ParamsDict()

    for shard_file, tensor_names in shard_files.items():
        shard_path = os.path.join(index_dir, shard_file)

        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        try:
            # Load the shard
            if framework == "pt":
                if not TORCH_AVAILABLE:
                    raise ImportError(
                        "PyTorch is not available. Cannot load with framework='pt'."
                    )

                from safetensors.torch import load_file as load_torch_file

                shard_data = load_torch_file(shard_path)
            else:
                if not NUMPY_AVAILABLE:
                    raise ImportError(
                        "NumPy is not available. Cannot load with framework='np'."
                    )

                from safetensors.numpy import load_file as load_numpy_file

                shard_data = load_numpy_file(shard_path)

            # Add the requested tensors to the params dict
            for tensor_name in tensor_names:
                if tensor_name in shard_data:
                    params_dict[tensor_name] = shard_data[tensor_name]
                else:
                    raise ValueError(
                        f"Tensor '{tensor_name}' not found in shard '{shard_file}'"
                    )

        except Exception as e:
            raise ValueError(f"Error loading shard '{shard_file}': {e}")

    return params_dict
