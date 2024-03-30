.PHONY: all run clean

all: run

run:
	python main.py

clean:
	rm -rf runs/*
	rm -rf tools/csv/*
	rm -rf tools/plot/*