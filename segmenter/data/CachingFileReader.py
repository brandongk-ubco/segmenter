import io
from typing import Dict


class CachingFileReader:

    cache: Dict[str, bytes] = {}

    def __init__(self, use_cache=True):
        self.use_cache = use_cache

    def read(self, filename):
        if filename not in self.cache:
            with open(filename, "rb") as newfile:
                result = newfile.read()
            if self.use_cache:
                self.cache[filename] = result
            return io.BytesIO(result)
        else:
            return io.BytesIO(self.cache[filename])
