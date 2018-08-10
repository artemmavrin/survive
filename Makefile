PYTHON := python3

IPYNB := $(shell find "examples" -name "*.ipynb" -maxdepth 1)

.PHONY: all install html test clean ipynb2rst

all: install html

install: clean
	${PYTHON} setup.py install

html: clean ipynb2rst
	make -C doc html

ipynb2rst: $(IPYNB)
	for f in examples/*.ipynb; do \
	  jupyter nbconvert --to rst $$f --output-dir=doc/source/examples/; \
	done;

test:
	${PYTHON} -m unittest discover --verbose

clean:
	rm -rf build dist *.egg-info doc/build doc/source/generated
	find . -name "__pycache__" -type d | xargs rm -rf
	find . -name ".ipynb_checkpoints" -type d | xargs rm -rf
	find . -name "*.pyc" -type f | xargs rm -f
