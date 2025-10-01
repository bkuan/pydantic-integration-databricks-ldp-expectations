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
    validators: List[str]
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
        
    def scan_repository(self, repo_path: str) -> Tuple[Dict[str, ModelInfo], Dict[str, ValidatorInfo]]:
        """
        Scan a repository for pydantic models and validators.
        
        Returns:
            Tuple of (models_dict, validators_dict)
        """
        repo_path = Path(repo_path).resolve()
        
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        self.logger.info(f"Scanning repository: {repo_path}")
        
        # Find all Python files
        python_files = self._find_python_files(repo_path)
        self.logger.info(f"Found {len(python_files)} Python files to scan")
        
        # Scan each file
        for file_path in python_files:
            try:
                self._scan_file(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to scan file {file_path}: {e}")
        
        self.logger.info(f"Discovery complete: {len(self.discovered_models)} models, "
                        f"{len(self.discovered_validators)} validators")
        
        return self.discovered_models, self.discovered_validators
    
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
        file_str = str(file_path)
        
        for pattern in self.config.exclude_patterns:
            # Convert glob pattern to simple string matching
            pattern_clean = pattern.replace("*/", "").replace("*", "")
            if pattern_clean in file_str:
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
    
    def _get_module_path(self, file_path: Path) -> str:
        """Convert file path to Python module path."""
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
            
            # Extract model information
            model_info = ModelInfo(
                name=node.name,
                module_path=module_path,
                file_path=str(file_path),
                class_def=node,
                fields=self._extract_model_fields(node),
                validators=self._extract_model_validators(node),
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
    
    def _extract_model_validators(self, class_def: ast.ClassDef) -> List[str]:
        """Extract validator function names from pydantic model."""
        validators = []
        
        for node in class_def.body:
            if isinstance(node, ast.FunctionDef):
                # Look for validator decorators
                for decorator in node.decorator_list:
                    decorator_name = self._get_decorator_name(decorator)
                    if decorator_name in {"validator", "field_validator", "model_validator"}:
                        validators.append(node.name)
                        break
        
        return validators
    
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
