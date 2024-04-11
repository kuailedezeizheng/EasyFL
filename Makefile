.PHONY: all run clean

all: run

run:
	python main.py

clean:
	rm -rf runs/*
	rm -rf result/csv/*
	rm -rf result/plot/*
	rm -rf result/time/*
	rm -rf ../../tf-logs/*