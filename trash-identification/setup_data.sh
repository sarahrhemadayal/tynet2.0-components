#!/bin/bash

# Create the main data directory
mkdir -p data/train/biodegradable data/train/recyclable data/train/hazardous
mkdir -p data/val/biodegradable data/val/recyclable data/val/hazardous

# Print the directory structure to verify
echo "Directory structure created:"
tree data
