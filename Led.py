from pyfirmata import Arduino, SERVO
import time
puerto = "\\.\COM5" #Puerto COM de emulación en USB
pin = (13) #PIN donde va conectado el LED

#Conexión con placa Arduino
print("Conectando con Arduino por USB...")
tarjeta = Arduino(puerto)
tarjeta.digital[pin].mode = SERVO
servo = tarjeta.get_pin('d:9:s')
pin.set_analog_period(20)  # Send pulses at 50 Hz
pin.write_analog(1023 * 1.2 / 20)
