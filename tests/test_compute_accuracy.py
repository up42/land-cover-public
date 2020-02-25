import sys
import os
import subprocess
from pathlib import Path
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

def test_compute_accuracy(capsys):
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        input_csv = mock_data / "test_extended-test_tiles.csv"
        out_file = mock_data / "m_3907833_nw_17_1_naip-new_class.tif"
        assert out_file.is_file()
        subprocess.run(
            "python3 landcover/compute_accuracy.py --output %s --input %s"
            % (str(mock_data), str(input_csv)),
            shell=True,
            check=True,
        )
        # Makes sure something is printed out
        captured = capsys.readouterr()
        assert captured
