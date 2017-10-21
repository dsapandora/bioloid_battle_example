#!/usr/bin/env pytho
import serial

ser = serial.Serial('/dev/ttyUSB0', 57600)
ser.write('WBLS\r')
print ser.read(4)
ser.close()
