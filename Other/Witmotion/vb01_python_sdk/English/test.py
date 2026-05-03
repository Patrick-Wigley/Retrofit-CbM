# ------------------------------------------------------
# test.py (English version)
# ------------------------------------------------------

import time
import device_model

"""
    WTVB01-485 Example
"""

# Common register address reference
"""
hex dec description

0x00   0   Save / Restart / Restore
0x04   4   UART baud rate
0x1A  26   Device address

0x3A  58   Vibration velocity X
0x3B  59   Vibration velocity Y
0x3C  60   Vibration velocity Z

0x3D  61   Vibration angle X
0x3E  62   Vibration angle Y
0x3F  63   Vibration angle Z

0x40  64   Temperature

0x41  65   Vibration displacement X
0x42  66   Vibration displacement Y
0x43  67   Vibration displacement Z

0x44  68   Vibration frequency X
0x45  69   Vibration frequency Y
0x46  70   Vibration frequency Z

0x63  99   Cutoff frequency
0x64 100   Cutoff frequency
0x65 101   Detection period
"""

# Create device model
device = device_model.DeviceModel("Test Device", "COM6", 9600, 0x50)

# Open device
device.openDevice()

# Start polling
device.startLoopRead()
time.sleep(0.5)

# Display data
while True:
    print(
        "vx:{} vy:{} vz:{} ax:{} ay:{} az:{} t:{} sx:{} sy:{} sz:{} fx:{} fy:{} fz:{}".format(
            device.get("58"), device.get("59"), device.get("60"),
            device.get("61"), device.get("62"), device.get("63"),
            device.get("64"), device.get("65"), device.get("66"),
            device.get("67"), device.get("68"), device.get("69"),
            device.get("70")
        )
    )
    time.sleep(0.2)

# Example manual register read:
# device.readReg(0x3A, 1)
# print(device.get(str(0x3A)))

# Example register write (set detection period to 50 Hz):
# device.writeReg(0x65, 50)
