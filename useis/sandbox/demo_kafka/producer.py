from confluent_kafka import Producer
import socket


def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg), str(err)))
    else:
        print("Message produced: %s" % (str(msg)))


conf = {'bootstrap.servers': "localhost:29092",
        'client.id': socket.gethostname()}

producer = Producer(conf)

p = Producer(conf)

p.produce('velocity', key="P", value="test", callback=acked)
p.flush()

p.poll(1)
