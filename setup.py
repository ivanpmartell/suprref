from setuptools import setup, find_packages

setup(
    #this will be the package name you will see, e.g. the output of 'conda list' in anaconda prompt
    name = 'suprref', 
    #some version number you may wish to add - increment this after every update
    version='0.1', 
  
    # Use one of the below approach to define package and/or module names:
  
    #if there are only handful of modules placed in root directory, and no packages/directories exist then can use below syntax
    # packages=[''], #have to import modules directly in code after installing this wheel, like import mod2 (respective file name in this case is mod2.py) - no direct use of distribution name while importing
  
    #can list down each package names - no need to keep __init__.py under packages / directories
    # packages=['<list of name of packages>'], #importing is like: from package1 import mod2, or import package1.mod2 as m2
  
    #this approach automatically finds out all directories (packages) - those must contain a file named __init__.py (can be empty)
    packages = find_packages(".")
    + find_packages(where="./CNNPROM")
    + find_packages(where="./IO")
    + find_packages(where="./modules"), #include/exclude arguments take * as wildcard, . for any sub-package names
    install_requires = ['requests', 'torch', 'tqdm', 'skorch', 'pandas', 'biopython', 'pyfaidx', 'PyInquirer', 'pyfiglet', 'mysql-connector-python'],
    entry_points = {
        'console_scripts': ['suprref=suprref.__main__:main'],
    }
)