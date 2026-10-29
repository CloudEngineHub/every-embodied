"""Render DexLab records on Windows using VTK's off-screen OpenGL window."""
import argparse
from pathlib import Path

import vtk

if not hasattr(vtk, "vtkWin32OpenGLRenderWindow"):
    raise SystemExit("This adapter requires Windows VTK with vtkWin32OpenGLRenderWindow.")
vtk.vtkEGLRenderWindow = vtk.vtkWin32OpenGLRenderWindow

from dexlab.cutting_showcase import render

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--stride", type=int, default=5)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--preview", action="store_true")
args = parser.parse_args()
if args.stride < 1 or args.fps < 1:
    parser.error("--stride and --fps must be positive")
render(args.directory, args.output, stride=args.stride, fps=args.fps, preview=args.preview)
