import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Now run the tests
from test_metrics import main

if __name__ == "__main__":
    main() 