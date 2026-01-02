"""
Model discovery module for scanning pydantic repositories.
"""

import ast
import inspect
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from .config import Config


@dataclass
class ModelInfo:
    """Information about a discovered pydantic model."""
    name: str
    module_path: str
    file_path: str
    class_def: ast.ClassDef
    fields: Dict[str, Any]
    validators: List[str]  # All validators (for backward compatibility)
    field_validators: Dict[str, List[str]]  # field_name -> [validator_method_names]
    model_validators: List[str]  # model-level validator method names
    base_classes: List[str]
    imports: Dict[str, str]  # import_name -> module_path
    
    @property
    def full_name(self) -> str:
        """Get fully qualified model name."""
        return f"{self.module_path}.{self.name}"


@dataclass
class ValidatorInfo:
    """Information about a discovered validator function."""
    name: str
    module_path: str
    file_path: str
    function_def: ast.FunctionDef
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]


class ModelDiscovery:
    """Discovers pydantic models and validators in source code repositories."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.discovered_models: Dict[str, ModelInfo] = {}
        self.discovered_validators: Dict[str, ValidatorInfo] = {}
        
    def scan_repository(self, repo_path: str = None) -> Tuple[Dict[str, ModelInfo], Dict[str, ValidatorInfo]]:
        """
        Scan a repository for pydantic models and validators.
        
        Args:
            repo_path: Optional path to scan. If None, scans all configured source_paths.
        
        Returns:
            Tuple of (models_dict, validators_dict)
        """
        if repo_path is not None:
            # Legacy behavior - scan single path
            return self._scan_single_repository(repo_path)
        
        # New behavior - scan all configured source paths
        return self.scan_all_repositories()
    
    def scan_all_repositories(self) -> Tuple[Dict[str, ModelInfo], Dict[str, ValidatorInfo]]:
        """
        Scan all configured source paths for pydantic models and validators.
        
        Returns:
            Tuple of (models_dict, validators_dict)
        """
        if not self.config.source_paths:
            raise ValueError("No source paths configured in config.source_paths")
        
        self.logger.info(f"Scanning {len(self.config.source_paths)} configured source paths")
        
        # Reset discovered items for fresh scan
        self.discovered_models = {}
        self.discovered_validators = {}
        
        # Scan each configured path
        for source_path in self.config.source_paths:
            try:
                self.logger.info(f"Scanning source path: {source_path}")
                models, validators = self._scan_single_repository(source_path)
                
                # Merge results (models and validators have unique full names)
                self.discovered_models.update(models)
                self.discovered_validators.update(validators)
                
                self.logger.info(f"  Found {len(models)} models, {len(validators)} validators")
                
            except Exception as e:
                self.logger.warning(f"Failed to scan source path {source_path}: {e}")
        
        self.logger.info(f"Total discovery complete: {len(self.discovered_models)} models, "
                        f"{len(self.discovered_validators)} validators")
        
        return self.discovered_models, self.discovered_validators
    
    def _scan_single_repository(self, repo_path: str) -> Tuple[Dict[str, ModelInfo], Dict[str, ValidatorInfo]]:
        """
        Scan a single repository path for pydantic models and validators.
        
        Args:
            repo_path: Path to scan
            
        Returns:
            Tuple of (models_dict, validators_dict) for this path only
        """
        repo_path = Path(repo_path).resolve()
        
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        # Create temporary storage for this scan
        original_models = self.discovered_models.copy()
        original_validators = self.discovered_validators.copy()
        
        # Clear for clean scan
        self.discovered_models = {}
        self.discovered_validators = {}
        
        try:
            self.logger.debug(f"Scanning single repository: {repo_path}")
            
            # Find all Python files
            python_files = self._find_python_files(repo_path)
            self.logger.debug(f"Found {len(python_files)} Python files to scan")
            
            # Scan each file
            for file_path in python_files:
                try:
                    self._scan_file(file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to scan file {file_path}: {e}")
            
            # Capture results for this scan
            path_models = self.discovered_models.copy()
            path_validators = self.discovered_validators.copy()
            
            return path_models, path_validators
            
        finally:
            # Restore original state
            self.discovered_models = original_models
            self.discovered_validators = original_validators
    
    def _find_python_files(self, repo_path: Path) -> List[Path]:
        """Find all Python files in repository, respecting exclusion patterns."""
        python_files = []
        
        for file_path in repo_path.rglob("*.py"):
            # Check exclusion patterns
            if self._should_exclude_file(file_path):
                continue
            
            python_files.append(file_path)
        
        return python_files
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded based on patterns."""
        import fnmatch
        
        file_str = str(file_path)
        
        for pattern in self.config.exclude_patterns:
            # Use proper glob matching for full path patterns
            if fnmatch.fnmatch(file_str, pattern):
                return True
                
            # For filename-only patterns (no path separators), check against filename
            if "/" not in pattern and fnmatch.fnmatch(file_path.name, pattern):
                return True
                
            # For patterns like "*/test_*", check if any path component matches "test_*"
            if pattern.startswith("*/") and not pattern.endswith("/*"):
                filename_pattern = pattern[2:]  # Remove "*/"
                if fnmatch.fnmatch(file_path.name, filename_pattern):
                    return True
        
        return False
    
    def _scan_file(self, file_path: Path) -> None:
        """Scan a single Python file for models and validators."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, IOError) as e:
            self.logger.warning(f"Could not read file {file_path}: {e}")
            return
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in file {file_path}: {e}")
            return
        
        # Extract module path
        module_path = self._get_module_path(file_path)
        
        # Analyze imports
        imports = self._extract_imports(tree)
        
        # Find models and validators
        self._extract_models(tree, file_path, module_path, imports)
        self._extract_validators(tree, file_path, module_path)
    
    def _get_module_path(self, file_path) -> str:
        """Convert file path to Python module path."""
        # Ensure file_path is a Path object
        if isinstance(file_path, str):
            from pathlib import Path
            file_path = Path(file_path)
        
        # Remove .py extension and convert path separators to dots
        parts = file_path.with_suffix('').parts
        
        # Find the start of the module path (skip common prefixes)
        start_index = 0
        for i, part in enumerate(parts):
            if part in {'src', 'lib', 'app'} or part.endswith('_lib'):
                start_index = i + 1
                break
        
        module_parts = parts[start_index:]
        return '.'.join(module_parts) if module_parts else file_path.stem
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, str]:
        """Extract import statements from AST."""
        imports = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = alias.name
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        name = alias.asname or alias.name
                        imports[name] = f"{node.module}.{alias.name}"
        
        return imports
    
    def _extract_models(self, tree: ast.AST, file_path: Path, module_path: str, imports: Dict[str, str]) -> None:
        """Extract pydantic models from AST."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            
            # Check if class inherits from pydantic BaseModel
            if not self._is_pydantic_model(node, imports):
                continue
            
            # Extract validator information
            all_validators, field_validators, model_validators = self._extract_model_validators(node)
            
            # Extract model information
            model_info = ModelInfo(
                name=node.name,
                module_path=module_path,
                file_path=str(file_path),
                class_def=node,
                fields=self._extract_model_fields(node),
                validators=all_validators,
                field_validators=field_validators,
                model_validators=model_validators,
                base_classes=[self._get_base_class_name(base, imports) for base in node.bases],
                imports=imports
            )
            
            self.discovered_models[model_info.full_name] = model_info
            self.logger.debug(f"Discovered model: {model_info.full_name}")
    
    def _is_pydantic_model(self, class_def: ast.ClassDef, imports: Dict[str, str]) -> bool:
        """Check if class definition represents a pydantic model."""
        for base in class_def.bases:
            base_name = self._get_base_class_name(base, imports)
            
            # Check against configured base classes
            for model_base in self.config.model_base_classes:
                if base_name == model_base or base_name.endswith(f".{model_base}"):
                    return True
        
        return False
    
    def _get_base_class_name(self, base: ast.expr, imports: Dict[str, str]) -> str:
        """Get the full name of a base class from AST node."""
        if isinstance(base, ast.Name):
            # Simple name like BaseModel
            name = base.id
            return imports.get(name, name)
        
        elif isinstance(base, ast.Attribute):
            # Attribute access like pydantic.BaseModel
            return ast.unparse(base)
        
        return str(base)
    
    def _extract_model_fields(self, class_def: ast.ClassDef) -> Dict[str, Any]:
        """Extract field definitions from pydantic model."""
        fields = {}
        
        for node in class_def.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                # Annotated assignment (field with type hint)
                field_name = node.target.id
                field_type = ast.unparse(node.annotation)
                field_default = ast.unparse(node.value) if node.value else None
                
                fields[field_name] = {
                    "type": field_type,
                    "default": field_default,
                    "annotation": node.annotation
                }
        
        return fields
    
    def _extract_model_validators(self, class_def: ast.ClassDef) -> tuple[List[str], Dict[str, List[str]], List[str]]:
        """
        Extract validator function names from pydantic model.
        
        Returns:
            Tuple of (all_validators, field_validators_dict, model_validators_list)
        """
        all_validators = []
        field_validators = {}
        model_validators = []
        
        for node in class_def.body:
            if isinstance(node, ast.FunctionDef):
                # Look for validator decorators
                for decorator in node.decorator_list:
                    decorator_name = self._get_decorator_name(decorator)
                    
                    if decorator_name in {"validator", "field_validator"}:
                        # Field validator - extract field names from decorator
                        field_names = self._extract_field_names_from_decorator(decorator)
                        all_validators.append(node.name)
                        
                        for field_name in field_names:
                            if field_name not in field_validators:
                                field_validators[field_name] = []
                            field_validators[field_name].append(node.name)
                        break
                        
                    elif decorator_name == "model_validator":
                        # Model validator
                        all_validators.append(node.name)
                        model_validators.append(node.name)
                        break
        
        return all_validators, field_validators, model_validators
    
    def _extract_field_names_from_decorator(self, decorator: ast.expr) -> List[str]:
        """Extract field names from @field_validator decorator."""
        field_names = []
        
        if isinstance(decorator, ast.Call):
            # @field_validator('field_name') or @field_validator('field1', 'field2')
            for arg in decorator.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    field_names.append(arg.value)
                elif isinstance(arg, ast.Str):  # For older Python versions
                    field_names.append(arg.s)
        
        # If no field names found in decorator args, try to infer from method name
        if not field_names:
            # This is a fallback - in real usage, field_validator should specify fields
            pass
            
        return field_names
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return ""
    
    def _extract_validators(self, tree: ast.AST, file_path: Path, module_path: str) -> None:
        """Extract standalone validator functions from AST."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            
            # Check if function looks like a validator
            if not node.name.endswith("_validator"):
                continue
            
            validator_info = ValidatorInfo(
                name=node.name,
                module_path=module_path,
                file_path=str(file_path),
                function_def=node,
                parameters=[arg.arg for arg in node.args.args],
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=ast.get_docstring(node)
            )
            
            full_name = f"{module_path}.{node.name}"
            self.discovered_validators[full_name] = validator_info
            self.logger.debug(f"Discovered validator: {full_name}")
    
    def get_models_by_module(self, module_pattern: str) -> Dict[str, ModelInfo]:
        """Get models filtered by module pattern."""
        filtered_models = {}
        
        for full_name, model_info in self.discovered_models.items():
            if module_pattern in model_info.module_path:
                filtered_models[full_name] = model_info
        
        return filtered_models
    
    def get_model_dependencies(self, model_name: str) -> Set[str]:
        """Get dependencies for a specific model."""
        dependencies = set()
        
        if model_name not in self.discovered_models:
            return dependencies
        
        model_info = self.discovered_models[model_name]
        
        # Check field types for references to other models
        for field_name, field_info in model_info.fields.items():
            field_type = field_info["type"]
            
            # Look for other discovered models in field types
            for other_model in self.discovered_models:
                model_class_name = other_model.split(".")[-1]
                if model_class_name in field_type:
                    dependencies.add(other_model)
        
        return dependencies
