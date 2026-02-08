import serial
import sys

# Change to /dev/ttyACM1 if ACM0 doesn't show data
port = '/dev/ttyACM0' 
baud = 115200

try:
    # We open with a short timeout to prevent the script from hanging
    ser = serial.Serial(port, baud, timeout=1)
    print(f"--- Connected to {port} at {baud} ---")
    print("--- Press Ctrl+C to stop ---")

    while True:
        if ser.in_waiting > 0:
            # Read a line and decode it
            line = ser.readline().decode('utf-8', errors='replace').strip()
            if line:
                print(f"Received: {line}")
        
except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("\nClosing connection.")
    ser.close()
