@echo off
REM Setup script for LAKER-XSA development environment (Windows)

echo Setting up LAKER-XSA development environment...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install package in editable mode with development dependencies
echo Installing LAKER-XSA with development dependencies...
pip install -e ".[dev,bench,train]"

REM Run tests to verify installation
echo Running tests to verify installation...
pytest tests\ -v

echo.
echo Setup complete!
echo To activate the environment, run: venv\Scripts\activate
