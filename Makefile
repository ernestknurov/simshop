run:
	uv run python -m streamlit run src/interface/app.py --server.fileWatcherType none

train:
	uv run python -m src.scripts.train $(ARGS)

evaluate:
	uv run python -m src.scripts.evaluate $(ARGS)