from __future__ import unicode_literals, print_function

import os

# Make tmp directory for test
TMP_DIR = os.path.join(os.path.dirname(__file__), 'tmp')
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)


def file_path(file_name):
    """
    Get tmp file path
    """
    return os.path.join(TMP_DIR, file_name)
