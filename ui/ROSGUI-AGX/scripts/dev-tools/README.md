# Development Tools

This directory contains development and maintenance scripts for the ROSGUI project.

## Scripts

### Documentation Tools

- **fix_all_warnings.py** - Comprehensive script to fix all documentation warnings
- **fix_doc_warnings.py** - Automated title underline detection and fixing

## Usage

### Fix Documentation Warnings

```bash
# From project root
cd scripts/dev-tools
python3 fix_all_warnings.py

# Or from docs directory
cd docs
python3 ../scripts/dev-tools/fix_all_warnings.py
```

### Fix Specific Documentation Issues

```bash
cd scripts/dev-tools
python3 fix_doc_warnings.py
```

## Requirements

- Python 3.6+
- Access to the docs/ directory
- Write permissions for documentation files

## Notes

These scripts are designed to maintain documentation quality and should be run before committing documentation changes.
