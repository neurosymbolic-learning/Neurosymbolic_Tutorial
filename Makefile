
all: webbook/notebooks/neurosymbolic_notebook1.ipynb webbook/notebooks/neurosymbolic_notebook2.ipynb webbook/notebooks/neurosymbolic_notebook3.ipynb webbook/notebooks/imgs webbook/README.md
	jupyter-book build webbook	

webbook/notebooks/neurosymbolic_notebook1.ipynb:
	jq -M 'del(.metadata.widgets)' neurosymbolic_notebook1.ipynb > webbook/notebooks/neurosymbolic_notebook1.ipynb

webbook/notebooks/neurosymbolic_notebook2.ipynb:
	jq -M 'del(.metadata.widgets)' neurosymbolic_notebook2.ipynb > webbook/notebooks/neurosymbolic_notebook2.ipynb

webbook/notebooks/neurosymbolic_notebook3.ipynb:
	jq -M 'del(.metadata.widgets)' neurosymbolic_notebook3.ipynb > webbook/notebooks/neurosymbolic_notebook3.ipynb

webbook/notebooks/imgs:
	cp -r imgs webbook/notebooks/imgs

webbook/README.md:
	cp README.md webbook/README.md

clean:
	rm -f webbook/notebooks/neurosymbolic_notebook1.ipynb
	rm -f webbook/notebooks/neurosymbolic_notebook2.ipynb
	rm -f webbook/notebooks/neurosymbolic_notebook3.ipynb
	rm -rf webbook/notebooks/imgs
	rm -f webbook/README.md
	jupyter-book clean webbook --all
