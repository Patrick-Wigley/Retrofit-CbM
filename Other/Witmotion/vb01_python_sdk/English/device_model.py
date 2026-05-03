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
    auchCRCHi = [ 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
        0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
        0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01,
        0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
        0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81,
        0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
        0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01,
        0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
        0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
        0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
        0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01,
        0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
        0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
        0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
        0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01,
        0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
        0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
        0x40]  # unchanged lookup table
    auchCRCLo = [ 0x00, 0xC0, 0xC1, 0x01, 0xC3, 0x03, 0x02, 0xC2, 0xC6, 0x06, 0x07, 0xC7, 0x05, 0xC5, 0xC4,
        0x04, 0xCC, 0x0C, 0x0D, 0xCD, 0x0F, 0xCF, 0xCE, 0x0E, 0x0A, 0xCA, 0xCB, 0x0B, 0xC9, 0x09,
        0x08, 0xC8, 0xD8, 0x18, 0x19, 0xD9, 0x1B, 0xDB, 0xDA, 0x1A, 0x1E, 0xDE, 0xDF, 0x1F, 0xDD,
        0x1D, 0x1C, 0xDC, 0x14, 0xD4, 0xD5, 0x15, 0xD7, 0x17, 0x16, 0xD6, 0xD2, 0x12, 0x13, 0xD3,
        0x11, 0xD1, 0xD0, 0x10, 0xF0, 0x30, 0x31, 0xF1, 0x33, 0xF3, 0xF2, 0x32, 0x36, 0xF6, 0xF7,
        0x37, 0xF5, 0x35, 0x34, 0xF4, 0x3C, 0xFC, 0xFD, 0x3D, 0xFF, 0x3F, 0x3E, 0xFE, 0xFA, 0x3A,
        0x3B, 0xFB, 0x39, 0xF9, 0xF8, 0x38, 0x28, 0xE8, 0xE9, 0x29, 0xEB, 0x2B, 0x2A, 0xEA, 0xEE,
        0x2E, 0x2F, 0xEF, 0x2D, 0xED, 0xEC, 0x2C, 0xE4, 0x24, 0x25, 0xE5, 0x27, 0xE7, 0xE6, 0x26,
        0x22, 0xE2, 0xE3, 0x23, 0xE1, 0x21, 0x20, 0xE0, 0xA0, 0x60, 0x61, 0xA1, 0x63, 0xA3, 0xA2,
        0x62, 0x66, 0xA6, 0xA7, 0x67, 0xA5, 0x65, 0x64, 0xA4, 0x6C, 0xAC, 0xAD, 0x6D, 0xAF, 0x6F,
        0x6E, 0xAE, 0xAA, 0x6A, 0x6B, 0xAB, 0x69, 0xA9, 0xA8, 0x68, 0x78, 0xB8, 0xB9, 0x79, 0xBB,
        0x7B, 0x7A, 0xBA, 0xBE, 0x7E, 0x7F, 0xBF, 0x7D, 0xBD, 0xBC, 0x7C, 0xB4, 0x74, 0x75, 0xB5,
        0x77, 0xB7, 0xB6, 0x76, 0x72, 0xB2, 0xB3, 0x73, 0xB1, 0x71, 0x70, 0xB0, 0x50, 0x90, 0x91,
        0x51, 0x93, 0x53, 0x52, 0x92, 0x96, 0x56, 0x57, 0x97, 0x55, 0x95, 0x94, 0x54, 0x9C, 0x5C,
        0x5D, 0x9D, 0x5F, 0x9F, 0x9E, 0x5E, 0x5A, 0x9A, 0x9B, 0x5B, 0x99, 0x59, 0x58, 0x98, 0x88,
        0x48, 0x49, 0x89, 0x4B, 0x8B, 0x8A, 0x4A, 0x4E, 0x8E, 0x8F, 0x4F, 0x8D, 0x4D, 0x4C, 0x8C,
        0x44, 0x84, 0x85, 0x45, 0x87, 0x47, 0x46, 0x86, 0x82, 0x42, 0x43, 0x83, 0x41, 0x81, 0x80,
        0x40]  # unchanged lookup table
    # endregion

    def __init__(self, deviceName, portName, baud, ADDR):
        print("Initializing device model")
        self.deviceName = deviceName
        self.serialConfig.portName = portName
        self.serialConfig.baud = baud
        self.ADDR = ADDR

    # Calculate CRC
    def get_crc(self, datas, dlen):
        tempH = 0xff  # 高 CRC 字节初始化 High CRC byte initialization
        tempL = 0xff  # 低 CRC 字节初始化 Low CRC byte initialization
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
                # Set each value found at its corresponding register
                self.set(str(self.statReg), value)
                self.statReg += 1
            self.TempBytes.clear()

    # Send raw bytes
    def sendData(self, data):
        DEBUG = False
        try:
            if DEBUG:
                print(f"Sending: ", [hex(i) for i in data])
                print(data)
            self.serialPort.write(data) # Dont specify size of buffer in py
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


