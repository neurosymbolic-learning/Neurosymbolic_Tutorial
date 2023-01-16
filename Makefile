# This makefile generates the webbook for the Neurosymbolic notebook
# Usage: 
# make all: generates the webbook 
# make clean: cleans the webbook and all generated files
# requires jq and jupyter-book


all: webbook

transfer: webbook
	cp -r webbook/* ../popl23tutorial	

webbook: webbook/neurosymbolic_notebook1.ipynb webbook/neurosymbolic_notebook2.ipynb webbook/neurosymbolic_notebook3.ipynb webbook/README.md
	jupyter-book build webbook

webbook/neurosymbolic_notebook1.ipynb:
	jq -M 'del(.metadata.widgets)' neurosymbolic_notebook1.ipynb > webbook/neurosymbolic_notebook1.ipynb

webbook/neurosymbolic_notebook2.ipynb:
	jq -M 'del(.metadata.widgets)' neurosymbolic_notebook2.ipynb > webbook/neurosymbolic_notebook2.ipynb

webbook/neurosymbolic_notebook3.ipynb:
	jq -M 'del(.metadata.widgets)' neurosymbolic_notebook3.ipynb > webbook/neurosymbolic_notebook3.ipynb

webbook/README.md:
	cp README.md webbook/README.md

clean:
	rm -f webbook/neurosymbolic_notebook1.ipynb
	rm -f webbook/neurosymbolic_notebook2.ipynb
	rm -f webbook/neurosymbolic_notebook3.ipynb
	rm -f webbook/README.md
	jupyter-book clean webbook --all
