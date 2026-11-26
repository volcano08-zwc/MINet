"""Compatibility entry point for full Session 1 training. Use train_2a.py."""
import sys
from train_2a import main

if __name__ == "__main__":
    sys.argv.extend(["--protocol", "full-session1"])
    main()
