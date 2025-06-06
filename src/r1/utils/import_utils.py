# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utility module
Check if optional dependencies are available
"""

import importlib.util


def is_package_available(package_name: str) -> bool:
    """Check if package is available"""
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def is_e2b_available() -> bool:
    """Check if e2b-code-interpreter is available"""
    return is_package_available("e2b_code_interpreter")


def is_morph_available() -> bool:
    """Check if morphcloud is available"""
    return is_package_available("morphcloud")


def is_wandb_available() -> bool:
    """Check if wandb is available"""
    return is_package_available("wandb")


def is_torch_available() -> bool:
    """Check if torch is available"""
    return is_package_available("torch")


def is_transformers_available() -> bool:
    """Check if transformers is available"""
    return is_package_available("transformers")


def is_trl_available() -> bool:
    """Check if trl is available"""
    return is_package_available("trl")


def check_required_packages():
    """Check if all required packages are installed"""
    required_packages = {
        "torch": is_torch_available(),
        "transformers": is_transformers_available(),
    }
    
    missing_packages = [pkg for pkg, available in required_packages.items() if not available]
    
    if missing_packages:
        raise ImportError(f"Missing required packages: {', '.join(missing_packages)}")
    
    return True 