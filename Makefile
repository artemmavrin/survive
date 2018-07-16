PYTHON := python3

.PHONY: install test clean

install: clean
	${PYTHON} setup.py install

test:
	${PYTHON} -m unittest discover --verbose

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d | xargs rm -rf
	find . -name ".ipynb_checkpoints" -type d | xargs rm -rf
	find . -name "*.pyc" -type f | xargs rm -f
	find . -type d -empty -delete
