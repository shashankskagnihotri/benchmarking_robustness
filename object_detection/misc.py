import re
from fractions import Fraction
from pathlib import Path


def find_latest_epoch_file(directory):
    root_dir = Path(directory)
    max_num = -1
    latest_file = None

    # Regex to match files of the form 'epoch_{n}.pth' where {n} is an integer
    pattern = re.compile(r"^epoch_(\d+)\.pth$")

    # Find the file with the highest epoch number
    for file in root_dir.iterdir():
        match = pattern.match(file.name)
        if match:
            current_num = int(match.group(1))
            if current_num > max_num:
                max_num = current_num
                latest_file = file

    # check if there is a single *.pth file
    if not latest_file:
        pth_files = list(root_dir.glob("*.pth"))
        if len(pth_files) == 1:
            return pth_files[0]
        else:
            return None

    return latest_file


def find_python_files(directory):
    python_files = list(directory.glob("*.py"))  # Glob for Python files only

    if python_files:
        assert len(python_files) == 1, f"Multiple Python files found in {directory}"
        return python_files[0]
    else:
        return None


def format_value(v):
    if isinstance(v, float):
        fraction = Fraction(v).limit_denominator(1000)
        if fraction.denominator <= 255:
            return f"{fraction.numerator}div{fraction.denominator}"
        else:
            return f"{v:.2f}"  # Limit to 2 decimal places
    return str(v)
