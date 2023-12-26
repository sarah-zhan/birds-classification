FROM public.ecr.aws/lambda/python:3.9

RUN pip install keras-image-helper
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

COPY xception_v4_36_0.929.tflite .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]