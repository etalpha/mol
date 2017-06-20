from distutils.core import setup
import os

def is_package(path):
    base = os.path.basename(path)
    return (os.path.isdir(path)
            and not 
            (base[:2] == '__' and base[-2:] == '__')
            and base[0] != '.'
           )

def all_dir(path):
    join = os.path.join
    return [join(path, p) for p in os.listdir(path) if is_package(join(path, p))]

def directories(path):
    dirs = all_dir(path)
    for d in dirs[:]:
        dirs.extend(directories(d))
    return dirs
    
def packages(path):
    "search packages recursionary"
    packs = directories(path)
    return [path] + [p.replace('/', '.') for p in packs]


setup(
        name='mol',
        packages=packages('mol'),
        version='0.0.0',
        )
