# ------------------------------------------------------
# # device_model.py (English version)
# ------------------------------------------------------




# coding:UTF-8
import threading
import time
import serial
from serial import SerialException


# Serial port configuration
class SerialConfig:
    # Serial port name
    portName = ''

    # Baud rate
    baud = 9600


# Device instance
class DeviceModel:
    # region Properties

    # Device name
    deviceName = "My Device"

    # Device Modbus ID
    ADDR = 0x50

    # Device data dictionary
    deviceData = {}

    # Device open status
    isOpen = False

    # Loop read flag
    loop = False

    # Serial port
    serialPort = None

    # Serial port config
    serialConfig = SerialConfig()

    # Temporary buffer
    TempBytes = []

    # Starting register
    statReg = None

    # endregion

    # region CRC calculation tables
    auchCRCHi = [...]  # unchanged lookup table
    auchCRCLo = [...]  # unchanged lookup table
    # endregion

    def __init__(self, deviceName, portName, baud, ADDR):
        print("Initializing device model")
        self.deviceName = deviceName
        self.serialConfig.portName = portName
        self.serialConfig.baud = baud
        self.ADDR = ADDR

    # Calculate CRC
    def get_crc(self, datas, dlen):
        tempH = 0xff  # High CRC byte init
        tempL = 0xff  # Low CRC byte init
        for i in range(0, dlen):
            tempIndex = (tempH ^ datas[i]) & 0xff
            tempH = (tempL ^ self.auchCRCHi[tempIndex]) & 0xff
            tempL = self.auchCRCLo[tempIndex]
        return (tempH << 8) | tempL

    # Set device data
    def set(self, key, value):
        self.deviceData[key] = value

    # Get device data
    def get(self, key):
        return self.deviceData.get(key, None)

    # Remove device data
    def remove(self, key):
        del self.deviceData[key]

    # Open device
    def openDevice(self):
        self.closeDevice()
        try:
            self.serialPort = serial.Serial(self.serialConfig.portName, self.serialConfig.baud, timeout=0.5)
            self.isOpen = True
            print(f"{self.serialConfig.portName} opened")
            # Start thread to read incoming data
            t = threading.Thread(target=self.readDataTh, args=("Data-Received-Thread", 10,))
            t.start()
            print("Device opened successfully")
        except SerialException:
            print(f"Failed to open {self.serialConfig.portName}")

    # Serial read thread
    def readDataTh(self, threadName, delay):
        print("Starting " + threadName)
        while True:
            if self.isOpen:
                try:
                    tLen = self.serialPort.inWaiting()
                    if tLen > 0:
                        data = self.serialPort.read(tLen)
                        self.onDataReceived(data)
                except Exception as ex:
                    print(ex)
            else:
                time.sleep(0.1)
                print("Serial port not open")
                break

    # Close device
    def closeDevice(self):
        if self.serialPort is not None:
            self.serialPort.close()
            print("Port closed")
        self.isOpen = False
        print("Device closed")

    # Process incoming serial data
    def onDataReceived(self, data):
        tempdata = bytes.fromhex(data.hex())
        for val in tempdata:
            self.TempBytes.append(val)
            if self.TempBytes[0] != self.ADDR:
                del self.TempBytes[0]
                continue
            if len(self.TempBytes) > 2:
                if not (self.TempBytes[1] == 0x03):
                    del self.TempBytes[0]
                    continue
                tLen = len(self.TempBytes)
                if tLen == self.TempBytes[2] + 5:
                    tempCrc = self.get_crc(self.TempBytes, tLen - 2)
                    if (tempCrc >> 8) == self.TempBytes[tLen - 2] and (tempCrc & 0xff) == self.TempBytes[tLen - 1]:
                        self.processData(self.TempBytes[2])
                    else:
                        del self.TempBytes[0]

    # Parse data packet
    def processData(self, length):
        if self.statReg is not None:
            for i in range(int(length / 2)):
                value = self.TempBytes[2 * i + 3] << 8 | self.TempBytes[2 * i + 4]
                if 0x3D <= self.statReg <= 0x3F:  # Angle
                    # NOTE Reading registers for vibration angle
                    """ From test.py reg addr ref
                        statReg dict key for self.deviceData[] will be:
                        0x3D  61   Vibration angle X
                        0x3E  62   Vibration angle Y
                        0x3F  63   Vibration angle Z"""
                    value = value / 32768 * 180
                elif self.statReg == 0x40:       # Temperature
                    # NOTE Don't think [WTVB01-485] has temp sensor: https://witmotion-sensor.com/products/vibration-sensors-ip67-waterproof-and-dustproof-for-motor-pump-vibration-monitoring?variant=44915903791301
                    value = value / 100
                # NOTE: This sensor is also supposed to collect vibration velocity & frequency
                # REFER TO REG ADDR REF IN test.py
                # start my code
                # elif 0x3A <= self.statReg <= 0x3C:
                #     print(f"Vibration Velocity: {value}")
                # end my code

                # NOTE come into self.set to set dictionary (0x3D be populated with Vibration angle X data hopefully)
                self.set(str(self.statReg), value)
                self.statReg += 1
            self.TempBytes.clear()

    # Send raw bytes
    def sendData(self, data):
        try:
            self.serialPort.write(data)
        except Exception as ex:
            print(ex)

    # Read a register
    def readReg(self, regAddr, regCount):
        self.statReg = regAddr
        self.sendData(self.get_readBytes(self.ADDR, regAddr, regCount))

    # Write a register
    def writeReg(self, regAddr, sValue):
        self.unlock()
        time.sleep(0.1)
        self.sendData(self.get_writeBytes(self.ADDR, regAddr, sValue))
        time.sleep(0.1)
        self.save()

    # Build read command
    def get_readBytes(self, devid, regAddr, regCount):
        tempBytes = [None] * 8
        tempBytes[0] = devid
        tempBytes[1] = 0x03
        tempBytes[2] = regAddr >> 8
        tempBytes[3] = regAddr & 0xff
        tempBytes[4] = regCount >> 8
        tempBytes[5] = regCount & 0xff
        tempCrc = self.get_crc(tempBytes, len(tempBytes) - 2)
        tempBytes[6] = tempCrc >> 8
        tempBytes[7] = tempCrc & 0xff
        return tempBytes

    # Build write command
    def get_writeBytes(self, devid, regAddr, sValue):
        tempBytes = [None] * 8
        tempBytes[0] = devid
        tempBytes[1] = 0x06
        tempBytes[2] = regAddr >> 8
        tempBytes[3] = regAddr & 0xff
        tempBytes[4] = sValue >> 8
        tempBytes[5] = sValue & 0xff
        tempCrc = self.get_crc(tempBytes, len(tempBytes) - 2)
        tempBytes[6] = tempCrc >> 8
        tempBytes[7] = tempCrc & 0xff
        return tempBytes

    # Start loop reading
    def startLoopRead(self):
        self.loop = True
        t = threading.Thread(target=self.loopRead, args=())
        t.start()

    # Loop read thread
    def loopRead(self):
        print("Loop reading started")
        while self.loop:
            self.readReg(0x3A, 13)
            time.sleep(0.2)
        print("Loop reading stopped")

    # Stop loop reading
    def stopLoopRead(self):
        self.loop = False

    # Unlock config writes
    def unlock(self):
        cmd = self.get_writeBytes(self.ADDR, 0x69, 0xb588)
        self.sendData(cmd)

    # Save configuration
    def save(self):
        cmd = self.get_writeBytes(self.ADDR, 0x00, 0x0000)
        self.sendData(cmd)


