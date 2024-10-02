MAKEFILE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

FALDIR       = $(MAKEFILE_DIR)/projects/fal
FALCLIENTDIR = $(MAKEFILE_DIR)/projects/fal_client

.PHONY: docs

docs:
	$(MAKE) -C $(FALDIR) docs
	$(MAKE) -C $(FALCLIENTDIR) docs
	rm -rf docs
	mkdir -p docs/_build/html
	mkdir -p docs/_build/html/fal_sdk
	mkdir -p docs/_build/html/fal_client
	cp -a $(FALDIR)/docs/_build/html docs/_build/html/fal_sdk
	cp -a $(FALCLIENTDIR)/docs/_build/html docs/_build/html/fal_client
