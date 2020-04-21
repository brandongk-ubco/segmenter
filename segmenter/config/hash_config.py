import hashlib


def hash_config(in_string):
    return hashlib.md5(str(in_string).encode()).hexdigest()
