#!/usr/bin/env pytho
import serial

ser = serial.Serial('/dev/ttyUSB0', 57600)
ser.write('WFWD\r')
print ser.read(4)
ser.close()
