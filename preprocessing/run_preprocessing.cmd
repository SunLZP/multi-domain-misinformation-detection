@echo off
setlocal
set "EE_MODE=%~1"
if "%EE_MODE%"=="" set "EE_MODE=both"
if /I "%EE_MODE%"=="amt" goto find_python
if /I "%EE_MODE%"=="amt-splits" goto find_python
if /I "%EE_MODE%"=="multifc" goto find_python
if /I "%EE_MODE%"=="both" goto find_python
echo Usage: run_preprocessing.cmd [amt^|amt-splits^|multifc^|both]
exit /b 1

:find_python
set "EE_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "EE_PYARGS="
if not exist "%EE_PYTHON%" goto try_py
"%EE_PYTHON%" -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
if not errorlevel 1 goto validate_python
:try_py
py -3 -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
if not errorlevel 1 goto use_py
python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
if not errorlevel 1 goto use_python
echo [FAIL] No working Python 3.10+ found. Install Python or select a working interpreter in VS Code.
goto failed
:use_py
set "EE_PYTHON=py"
set "EE_PYARGS=-3"
goto validate_python
:use_python
set "EE_PYTHON=python"
:validate_python
"%EE_PYTHON%" %EE_PYARGS% -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
if errorlevel 1 goto failed
if /I "%EE_MODE%"=="multifc" goto multifc
if /I "%EE_MODE%"=="amt-splits" goto amt_splits
"%EE_PYTHON%" %EE_PYARGS% "%~dp0preprocess_amt.py"
if errorlevel 1 goto failed
if /I "%EE_MODE%"=="amt" goto done
:multifc
"%EE_PYTHON%" %EE_PYARGS% "%~dp0preprocess_multifc.py"
if errorlevel 1 goto failed
goto done
:amt_splits
"%EE_PYTHON%" %EE_PYARGS% "%~dp0preprocess_amt.py" --make-splits
if errorlevel 1 goto failed
:done
if "%~1"=="" pause
exit /b 0
:failed
echo [FAIL] Run did not complete. Read the error above; do not use results from a failed run.
if "%~1"=="" pause
exit /b 1
