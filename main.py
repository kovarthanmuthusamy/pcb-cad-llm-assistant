"""Launcher for ``python main.py``. The code lives in the ``pcb_assistant`` package:

pip install -r requirements.txt && pip install -e .
python main.py --provider ollama      # or: openai, google
# equivalent: python -m pcb_assistant --provider ollama
"""

from pcb_assistant.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
