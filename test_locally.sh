docker run \
	--rm \
	--volume .:/app \
	--volume /app/.venv \
	$(docker build -q .) \
	$@
