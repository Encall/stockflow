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
        
        # Execute the module
        spec.loader.exec_module(module)
        
        return True, "✅ Import OK"
    
    except ImportError as e:
        return False, f"❌ Import error: {str(e)}"
    except Exception as e:
        return False, f"❌ Execution error: {str(e)}"
    finally:
        # Clean up
        if spec and spec.name in sys.modules:
            del sys.modules[spec.name]


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
