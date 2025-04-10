# main.py
import os
import paho.mqtt.client as mqtt
from google.cloud import pubsub_v1
import time

def create_mqtt_client():
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(os.getenv('PROJECT_ID'), 'mqtt-data')

    def on_connect(client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        client.subscribe(os.getenv('MQTT_TOPIC', '#'))

    def on_message(client, userdata, msg):
        try:
            publisher.publish(topic_path, 
                            data=msg.payload,
                            mqtt_topic=msg.topic)
        except Exception as e:
            print(f"Publish error: {e}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Configure TLS if needed
    if os.getenv('MQTT_TLS') == 'true':
        client.tls_set()
    
    client.connect(os.getenv('MQTT_BROKER'), 
                  int(os.getenv('MQTT_PORT', 1883)), 
                  keepalive=60)
    return client

# Cloud Run requires an HTTP server
from flask import Flask
app = Flask(__name__)
client = None

@app.route('/')
def health_check():
    global client
    if not client or not client.is_connected():
        client = create_mqtt_client()
    client.loop_start()
    return "MQTT Bridge Running", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))