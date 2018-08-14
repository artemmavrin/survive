PYTHON := python3
PYLINT := pylint
PACKAGE := survive

.PHONY: all install html test clean lint

all: install html

install: clean
	${PYTHON} setup.py install

html: clean
	make -C doc html

test:
	${PYTHON} -m unittest discover --verbose

lint:
	${PYLINT} ${PACKAGE}

clean:
	rm -rf build dist *.egg-info doc/build doc/source/generated
	find . -name "__pycache__" -type d | xargs rm -rf
	find . -name ".ipynb_checkpoints" -type d | xargs rm -rf
	find . -name "*.pyc" -type f | xargs rm -f
