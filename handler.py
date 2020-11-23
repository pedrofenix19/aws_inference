import json
import functools
import tflite_runtime.interpreter as tflite
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import boto3
import botocore

#The following code is outside the handler because
#we want it to be executed just once when the Lambda 
#container is create and not in every Lambda execution

BUCKET_NAME = #Name of the public website S3 bucket
KEY = "model.tflite" #Name of the Tensorflow Lite model
LOCAL_MODEL_PATH = f"/tmp/{KEY}"

#The model is downloaded locally
s3 = boto3.resource('s3')

try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, LOCAL_MODEL_PATH)
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=LOCAL_MODEL_PATH)
interpreter.allocate_tensors()

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape((n_samples, latent_dim))
    return x_input

def generate_fake_samples():
    global interpreter
    latent_dim = 100
    n_samples = 25 #We will generate 25 images
    x_input = generate_latent_points(latent_dim, n_samples)


    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    outputs = []

    #We will pass each latent point to the model
    for i in x_input:
        i = np.expand_dims(i, axis=0)
        input_data = i
        input_data = input_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        outputs.append(interpreter.get_tensor(output_details[0]['index']))
    
    fake_samples = functools.reduce(lambda a, b: np.vstack((a,b)), outputs)

    return fake_samples

def get_images_base64(fake_samples):
    '''
        This function returns 25 base64 examples from fake_samples.
        fake_samples's shape must be (n_samples, width, height, 1)
    '''
    fig, ax = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].imshow(fake_samples[i * 5 + j,:,:,0], cmap='gray_r')

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    return base64.b64encode(pic_IObytes.read()).decode('ascii')
         

def get_image(event, context):
    #We use a plugin to keep the function warmed up so if this plugin is 
    #calling the function we do nothing
    if 'source' in event and event['source'] == "serverless-plugin-warmup":
        print(f"Warming up lambda")
        return{ "statusCode": 200, "body": "OK"}

    fake_samples = generate_fake_samples()
    img_base64 = get_images_base64(fake_samples)

    body = {
        "data": img_base64,
    }

    response = {
        "statusCode": 200,
        "headers": {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Credentials': "true",
        },
        "body": json.dumps(body)
    }

    return response

