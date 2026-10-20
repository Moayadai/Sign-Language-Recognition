import pathlib, sys
app_path = pathlib.Path('/run/media/moayad/19236c27-eb0e-4c6d-987b-16bf6f3051141/AI_Master/Ai/app.py')
lines = app_path.read_text().splitlines()
# Find the line with the print statement that indicates start of loop
start_idx = None
for i, line in enumerate(lines):
    if 'print(Show
