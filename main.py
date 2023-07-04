import numpy as np
import tensorflow as tf
import soundfile as sf
import pyaudio

audio = pyaudio.PyAudio()

CHUNK = 44032
RATE = 44100

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=1)

# Load the TFLite model and allocate tensors.
model_path = "./model/soundclassifier_with_metadata.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

frequency = 0

while (True):
    audio_data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
    desired_size = 44032  # 모델의 입력 크기에 맞춤
    if len(audio_data) > desired_size:
        audio_data = audio_data[:desired_size]
    elif len(audio_data) < desired_size:
        audio_data = np.pad(audio_data, (0, desired_size - len(audio_data)), 'constant')
    # Normalize audio data
    audio_data = audio_data.astype(np.float32) / 32768.0


    # Read the audio file and preprocess it.
    preprocessed_audio = np.expand_dims(audio_data, axis=0).astype(np.float32)

    # Set the preprocessed audio as input tensor.
    interpreter.set_tensor(input_details[0]['index'], preprocessed_audio)

    # Run the inference.
    interpreter.invoke()

    # Get the classification results.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    #print(output_data)
    print(np.argmax(output_data))
    if np.argmax(output_data) == 1:
        print("Go")
        break
    if np.argmax(output_data) == 0:
        print("Back")
        break
    if np.argmax(output_data) == 2:
        print("Left")
        break
    if np.argmax(output_data) == 3:
        print("Right")
        break


stream.stop_stream()
stream.close()
p.terminate()