"""
 Copyright (c) 2022, anonymous.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import io
import json
import logging
import os
import pickle
import re
import shutil
import urllib
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml
# from iopath.common.download import download
# from iopath.common.file_io import file_lock, g_pathmgr
from utils.registry import registry
from torch.utils.model_zoo import tqdm
# from torchvision.datasets.utils import (
#     check_integrity,
#     download_file_from_google_drive,
#     extract_archive,
# )



def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)
                    