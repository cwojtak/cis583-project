all:
	docker build . -t cis583-project
	docker run --name cis583-container --gpus all -it --rm -v -d \
		-v './models:/app/models' \
		-v './data:/app/data'  \
		-v './raw_results:/app/raw_results' \
		-v './test:/app/test' \
		cis583-project
