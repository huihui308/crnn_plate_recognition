"""
Patch for ONNX mapping module compatibility
This module provides the missing onnx.mapping functionality for compatibility
with older RKNN versions that expect this module to exist.
"""

import sys
import types

# Import onnx first
import onnx

# Check if onnx.mapping already exists
try:
    import onnx.mapping
    # If it already exists, no need for patching
except ImportError:
    # onnx.mapping doesn't exist, so we need to create it
    import numpy as np

    # Access the onnx module through globals to avoid local variable shadowing
    onnx_module = onnx

    # Create a mapping module that mimics the old API
    class MappingModule:
        def __init__(self):
            # Use the internal _mapping module to get the required attributes
            if hasattr(onnx_module, '_mapping'):
                # Copy the TENSOR_TYPE_MAP from _mapping
                self.TENSOR_TYPE_MAP = onnx_module._mapping.TENSOR_TYPE_MAP

                # Create TENSOR_TYPE_TO_NP_TYPE mapping based on the available mapping
                self.TENSOR_TYPE_TO_NP_TYPE = {}
                self.NP_TYPE_TO_TENSOR_TYPE = {}

                # Populate the mappings based on the TENSOR_TYPE_MAP
                for tensor_type, type_info in self.TENSOR_TYPE_MAP.items():
                    if hasattr(type_info, 'np_type'):
                        np_type = type_info.np_type
                        self.TENSOR_TYPE_TO_NP_TYPE[tensor_type] = np_type
                        self.NP_TYPE_TO_TENSOR_TYPE[np_type] = tensor_type
                    elif hasattr(type_info, 'np_dtype'):
                        np_type = type_info.np_dtype
                        self.TENSOR_TYPE_TO_NP_TYPE[tensor_type] = np_type
                        self.NP_TYPE_TO_TENSOR_TYPE[np_type] = tensor_type
            else:
                # Fallback: create basic mappings
                import onnx.numpy_helper

                # Basic tensor type to numpy type mapping
                self.TENSOR_TYPE_TO_NP_TYPE = {
                    1: np.float32,    # FLOAT
                    2: np.uint8,      # UINT8
                    3: np.int8,       # INT8
                    4: np.uint16,     # UINT16
                    5: np.int16,      # INT16
                    6: np.int32,      # INT32
                    7: np.int64,      # INT64
                    8: object,        # STRING
                    9: np.bool_,      # BOOL
                    10: np.float16,   # FLOAT16
                    11: np.float64,   # DOUBLE
                    12: np.uint32,    # UINT32
                    13: np.uint64,    # UINT64
                    14: np.complex64, # COMPLEX64
                    15: np.complex128 # COMPLEX128
                    # 16: np.bfloat16, # BFLOAT16 (not available in all numpy versions)
                }

                # Create reverse mapping
                self.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in self.TENSOR_TYPE_TO_NP_TYPE.items()}

                # Create a basic TENSOR_TYPE_MAP if needed
                self.TENSOR_TYPE_MAP = {}

    # Create an instance of the mapping module
    mapping_instance = MappingModule()

    # Add it to sys.modules so it can be imported as onnx.mapping
    onnx_mapping_module = types.ModuleType('onnx.mapping')

    # Copy all attributes from the mapping instance to the module
    for attr_name in dir(mapping_instance):
        if not attr_name.startswith('_') or attr_name in ['__dict__', '__doc__']:
            try:
                setattr(onnx_mapping_module, attr_name, getattr(mapping_instance, attr_name))
            except AttributeError:
                # Skip read-only attributes
                pass

    # Add the module to sys.modules
    sys.modules['onnx.mapping'] = onnx_mapping_module

    # Also add the mapping module as an attribute to the main onnx module
    # This is needed for some libraries that access onnx.mapping directly
    setattr(onnx_module, 'mapping', onnx_mapping_module)