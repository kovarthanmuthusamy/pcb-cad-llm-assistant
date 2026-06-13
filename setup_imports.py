import sys
import os

# Add RAGv2 to Python's module search path
ragv2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "RAGv2"))
if ragv2_path not in sys.path:
    sys.path.append(ragv2_path)