import re

splitter = re.compile(r"((?P<key>\w+)\s*=\s*(?P<val>(\s*[^ #!])*))?(?P<comment>(!|#).*)?")

def read_key_val_comment(line):
    match = splitter.match(line)
    assert (match is not None), f"line = {line}"

def read_tag(f):
    line = f.readline().strip()
    match = splitter.match(line)
    if match is not None:
        print("key: {}, val: {}, com: {}".format(match["key"], match["val"], match["comment"]))
    else:
        print("there is not tag")
