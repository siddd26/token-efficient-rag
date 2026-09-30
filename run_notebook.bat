@echo off
REM Optional helper: sets up a local Python environment and executes Token_Efficient_RAG.ipynb in place
REM (outputs, tables and charts are saved inside the notebook). You can also open the notebook in Jupyter/VS Code.
setlocal
cd /d "%~dp0"
echo.
echo  Token-Efficient RAG experiment
echo  ------------------------------
echo  1 = FULL experiment: preflight + tests + smoke test + 3 runs x 24 questions x 2 pipelines + analysis
echo      (about 60-90 minutes; LM Studio must be running with gemma-3-4b-it loaded)
echo  2 = ANALYSIS ONLY: recompute metrics, charts and report from existing results (no LLM calls, ~1 minute)
echo.
set /p CHOICE=Type 1 or 2 and press Enter: 
if "%CHOICE%"=="1" (set RAG_RUN_MODE=full) else (set RAG_RUN_MODE=analyze_existing)
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
if not exist .env (
  echo LLM_PROVIDER=lm_studio> .env
  echo OPENAI_BASE_URL=http://localhost:1234/v1>> .env
  echo OPENAI_API_KEY=lm-studio>> .env
  echo OPENAI_MODEL=gemma-3-4b-it>> .env
)
echo [1/3] Preparing Python environment (first time only; downloads packages)...
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe ( echo FAILED: could not create a Python virtual environment. & goto :end )
set PY=.venv\Scripts\python.exe
%PY% -m pip install --upgrade pip > setup_log.txt 2>&1
%PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu >> setup_log.txt 2>&1
%PY% -m pip install -r requirements.txt >> setup_log.txt 2>&1
if errorlevel 1 ( echo FAILED installing packages - see setup_log.txt & goto :end )
%PY% -m ipykernel install --sys-prefix --name token_efficient_rag --display-name "Token-Efficient RAG (.venv)" >> setup_log.txt 2>&1
echo [2/3] Executing the notebook in mode %RAG_RUN_MODE% - progress is written to results\run_log.txt ...
%PY% -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=token_efficient_rag Token_Efficient_RAG.ipynb > notebook_execution_log.txt 2>&1
if errorlevel 1 ( echo FAILED - see notebook_execution_log.txt and results\run_log.txt & goto :end )
echo [3/3] DONE. Open Token_Efficient_RAG.ipynb or results\experiment_report.md.
echo DONE > notebook_DONE.txt
:end
echo.
pause
