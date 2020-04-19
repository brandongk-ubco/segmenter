import hashlib


def hash(in_string):
    return hashlib.md5(str(in_string).encode()).hexdigest()
