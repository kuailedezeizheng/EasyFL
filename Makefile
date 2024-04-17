.PHONY: all run clean

all: alone

dir:
	mkdir -p result/csv result/plot result/time result/models result/poisoned_imgs

mul:
	python main.py

alone:
	python alone_main.py

clean:
	rm -rf runs/*
	rm -rf result/csv/*
	rm -rf result/plot/*
	rm -rf result/time/*
	rm -rf result/models/*
	rm -rf ../../tf-logs/*