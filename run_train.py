#!/usr/bin/env python
"""Simple wrapper to run training with correct imports."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train, parse_args

if __name__ == '__main__':
    args = parse_args()
    train(args)
