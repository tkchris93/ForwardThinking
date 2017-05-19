#!/bin/bash

echo "User note:  This test will use substantial computer memory (>50 GB)."
echo "If you do not have these memory resources on your machine, we advise you to cancel this task now."

echo ""
echo "Building 7x7 Window Features"
python mgs.py 7

echo "Building 9x9 Window Features"
python mgs.py 9

echo "Building 14x14 Window Features"
python mgs.py 14

echo "Concatenating Multi-Grained Scanning Output"
python concat_mgsout.py

echo "Building Foward Thinking Deep Random Forest (FTDRF)"
python FTDRF_from_mgs.py
