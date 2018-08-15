PYTHON := python3
PYLINT := pylint
SETUP := setup.py

PACKAGE := survive

.PHONY: all install html test clean lint

all: install html

install: clean
	${PYTHON} ${SETUP} install

html: clean
	mkdir -p doc/source/_static
	make -C doc html

test: #install
	${PYTHON} ${SETUP} test

lint:
	${PYLINT} ${PACKAGE}

clean:
	rm -rf build dist *.egg-info doc/build doc/source/generated
	find . -name "__pycache__" -type d | xargs rm -rf
	find . -name ".ipynb_checkpoints" -type d | xargs rm -rf
	find . -name "*.pyc" -type f | xargs rm -f
	find . -type d -empty -delete
