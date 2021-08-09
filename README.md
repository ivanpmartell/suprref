# SUpervised PRomoter REcognition Framework
For research on promoter recognition using deep learning.

## Docker installation
Requires docker or docker desktop.

### Windows:

```build-docker.bat```

Then open a command prompt (cmd.exe) and run the executable:

```suprref.bat```

### Linux/Unix:

```build-docker.sh```

Then open a terminal and run the executable:

```suprref.sh```

## Manual Installation
Requires python >3.6.9 and pip >21. First create the local wheel:

```python setup.py bdist_wheel```

Then install it:

```pip install dist/suprref-0.1-py3-none-any.whl```

This will create

This should automatically install any required python libraries that SUPRREF uses, including:

- numpy
- pytorch
- tqdm
- skorch
- pandas
- biopython
- pyfaidx
- PyInquirer
- pyfiglet
- mysql-connector-python

## Usage
After installation you can run the executable with arguments.

- Bring up the help manual
```suprref -h```

## Troubleshooting
This python framework has been tested on Ubuntu 18.04 (bionic) using the python3 and pip3 commands found in the default repositories.

Remember to run all commands inside the promoters (this repository) folder.

If you do not obtain the suprref executable with the manual installation, try using a virtual environment:

- Install virtualenv for the creation of virtual environments
```pip install virtualenv```

- Create the virtual environment
```virtualenv venv```

- Activate the virtual environment
```source venv/bin/activate```

And follow the manual installation instructions.