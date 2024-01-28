all:
	docker build . -t cis583-project
	docker run --name cis583-container --gpus all -it --rm -v -d cis583-project
