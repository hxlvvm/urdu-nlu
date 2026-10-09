FROM python:3.12-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY nlu/__init__.py nlu/data.py nlu/predictor.py nlu/
COPY serve/ serve/
# model.int8.onnx, tokenizer.json and labels.json (from the GitHub release, or `python -m nlu.export_onnx`)
COPY artifacts/model.int8.onnx artifacts/tokenizer.json artifacts/labels.json artifacts/
ENV NLU_ARTIFACTS=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
