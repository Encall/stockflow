#!/usr/bin/env python3
"""
Lightweight DAG validation script.
Validates Airflow DAGs without requiring full Airflow installation.
"""
import ast
import sys
import importlib.util
from pathlib import Path
from typing import Tuple, List
import types
import os


def validate_python_syntax(filepath: Path) -> Tuple[bool, str]:
    """Validate Python syntax using AST parsing."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read(), str(filepath))
        return True, "✅ Syntax OK"
    except SyntaxError as e:
        return False, f"❌ Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"❌ Parse error: {str(e)}"


def check_dag_structure(filepath: Path) -> Tuple[bool, str]:
    """Check for basic DAG structure requirements."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            tree = ast.parse(content, str(filepath))
        
        issues = []
        
        # Check for DAG import
        has_dag_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'airflow' in node.module:
                    for alias in node.names:
                        if alias.name in ('DAG', 'dag'):
                            has_dag_import = True
                            break
        
        if not has_dag_import:
            issues.append("No DAG import found")
        
        # Check for DAG instantiation or decorator
        has_dag_definition = False
        for node in ast.walk(tree):
            # Check for DAG() instantiation
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'DAG':
                    has_dag_definition = True
                    break
            # Check for @dag decorator
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'dag':
                        has_dag_definition = True
                        break
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name) and decorator.func.id == 'dag':
                            has_dag_definition = True
                            break
        
        if not has_dag_definition:
            issues.append("No DAG definition found (missing DAG() or @dag)")
        
        if issues:
            return False, f"❌ Structure issues: {', '.join(issues)}"
        
        return True, "✅ Structure OK"
    
    except Exception as e:
        return False, f"❌ Structure check failed: {str(e)}"


def test_dag_import(filepath: Path) -> Tuple[bool, str]:
    """Test if DAG file can be imported without errors."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"dag_module_{filepath.stem}", 
            filepath
        )
        if spec is None or spec.loader is None:
            return False, "❌ Could not create module spec"
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        
        # Before executing, protect imports that expect a running Airflow
        # environment by stubbing out `airflow` modules referenced by the file.
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                tree = ast.parse(content, str(filepath))

            # Find all imports that reference `airflow` and create lightweight
            # stub modules/attributes in sys.modules so importing the DAG won't
            # attempt to use a real Airflow DB, Variables, or provider hooks.
            airflow_imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('airflow'):
                        airflow_imports.append((node.module, [a.name for a in node.names]))
                if isinstance(node, ast.Import):
                    for n in node.names:
                        if n.name.startswith('airflow'):
                            airflow_imports.append((n.name, None))

            # Create stubs for each referenced airflow module and imported names
            created = []
            for module_name, names in airflow_imports:
                parts = module_name.split('.')
                for i in range(1, len(parts) + 1):
                    mod = '.'.join(parts[:i])
                    if mod in sys.modules:
                        continue
                    m = types.ModuleType(mod)
                    # Top-level `airflow` shim: provide models.Variable & DAG
                    if i == 1:
                        models_m = types.ModuleType('airflow.models')

                        class Variable:
                            @staticmethod
                            def get(key, default_var=None, deserialize_json=False):
                                # Prefer provided default_var, then environment variable
                                if default_var is not None:
                                    return default_var
                                return os.environ.get(key, '')

                        models_m.Variable = Variable
                        m.models = models_m
                        # Provide a minimal DAG class to allow DAG instantiation
                        class _FakeDAG:
                            def __init__(self, *a, **kw):
                                return None

                            def __enter__(self):
                                return self

                            def __exit__(self, exc_type, exc, tb):
                                return False

                        m.DAG = _FakeDAG
                    sys.modules[mod] = m
                    created.append(mod)

                # If this import requested specific names (from ... import X),
                # add those names to the module object so `from` imports succeed.
                if names:
                    target_mod = sys.modules.get(module_name)
                    if target_mod is None:
                        target_mod = types.ModuleType(module_name)
                        sys.modules[module_name] = target_mod
                        created.append(module_name)
                    class _DummyCallable:
                        def __init__(self, *a, **kw):
                            return None

                        def __call__(self, *a, **kw):
                            return None

                        def __rshift__(self, other):
                            return self

                        def __lshift__(self, other):
                            return self

                        def __rrshift__(self, other):
                            return self

                        def __rlshift__(self, other):
                            return self

                    for name in names:
                        if not hasattr(target_mod, name):
                            setattr(target_mod, name, _DummyCallable)

        except Exception:
            # Don't fail import stubbing on unexpected parsing errors – proceed
            created = []

        # Execute the module
        spec.loader.exec_module(module)
        
        return True, "✅ Import OK"
    
    except ImportError as e:
        return False, f"❌ Import error: {str(e)}"
    except Exception as e:
        return False, f"❌ Execution error: {str(e)}"
    finally:
        # Clean up the module we imported and any stubs we created.
        if spec and spec.name in sys.modules:
            del sys.modules[spec.name]
        # Remove any airflow.* stubs we added (leave real airflow packages intact)
        for mod in list(sys.modules.keys()):
            if mod == 'airflow' or mod.startswith('airflow.'):
                # Only remove entries we created during this import phase
                # Check for our simple attribute markers: models.Variable or a no-op DAG
                m = sys.modules.get(mod)
                try:
                    if (hasattr(m, 'models') and hasattr(m.models, 'Variable')) or hasattr(m, 'DAG'):
                        del sys.modules[mod]
                except Exception:
                    pass


def validate_dag_file(filepath: Path) -> bool:
    """Run all validations on a DAG file."""
    print(f"\n{'='*60}")
    try:
        display_path = filepath.relative_to(Path.cwd())
    except ValueError:
        display_path = filepath
    print(f"Validating: {display_path}")
    print(f"{'='*60}")
    
    all_passed = True
    
    # 1. Syntax validation
    passed, message = validate_python_syntax(filepath)
    print(f"  Syntax:    {message}")
    all_passed = all_passed and passed
    
    # 2. Structure validation
    passed, message = check_dag_structure(filepath)
    print(f"  Structure: {message}")
    all_passed = all_passed and passed
    
    # 3. Import validation (only if syntax and structure passed)
    if all_passed:
        passed, message = test_dag_import(filepath)
        print(f"  Import:    {message}")
        all_passed = all_passed and passed
    else:
        print(f"  Import:    ⏭️  Skipped (previous checks failed)")
    
    return all_passed


def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python validate_dags.py <dag_directory>")
        sys.exit(1)
    
    dag_dir = Path(sys.argv[1])
    
    if not dag_dir.exists():
        print(f"❌ Directory not found: {dag_dir}")
        sys.exit(1)
    
    # Find all Python files
    dag_files = list(dag_dir.glob("*.py"))
    
    if not dag_files:
        print(f"⚠️  No Python files found in {dag_dir}")
        sys.exit(0)
    
    print(f"\n🔍 Found {len(dag_files)} DAG file(s) to validate")
    
    results = []
    for dag_file in sorted(dag_files):
        # Skip __init__.py and test files
        if dag_file.name.startswith('__') or dag_file.name.startswith('test_'):
            print(f"\n⏭️  Skipping: {dag_file.name}")
            continue
        
        passed = validate_dag_file(dag_file)
        results.append((dag_file, passed))
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for dag_file, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {dag_file.name}")
    
    print(f"\nTotal: {passed_count}/{total_count} passed")
    
    if passed_count < total_count:
        print("\n❌ Some DAGs failed validation")
        sys.exit(1)
    else:
        print("\n✅ All DAGs passed validation")
        sys.exit(0)


if __name__ == "__main__":
    main()
