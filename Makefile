PYTHON := python3

.PHONY: all install html test clean ipynb2rst

all: install html

install: clean
	${PYTHON} setup.py install

html: clean ipynb2rst
	make -C doc html

test:
	${PYTHON} -m unittest discover --verbose

clean:
	rm -rf build dist *.egg-info doc/build doc/source/generated
	find . -name "__pycache__" -type d | xargs rm -rf
	find . -name ".ipynb_checkpoints" -type d | xargs rm -rf
	find . -name "*.pyc" -type f | xargs rm -f
