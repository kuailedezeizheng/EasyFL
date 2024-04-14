.PHONY: all run clean

all: run

mkdir:
	mkdir result/csv
	mkdir result/plot
	mkdir result/time
	mkdir poisoned_imgs

run:
	python main.py

alone:
	python alone_main.py

clean:
	rm -rf runs/*
	rm -rf result/csv/*
	rm -rf result/plot/*
	rm -rf result/time/*
	rm -rf ../../tf-logs/*