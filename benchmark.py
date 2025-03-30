
# import requests

# import subprocess
# subprocess.run(["curl", "-X", "GET", "http://localhost:8001/"])
# exit(0)


# import requests

# import socket
# socket.setdefaulttimeout(10)
# # Force IPv4
# import urllib3
# urllib3.util.connection.HAS_IPV6 = False

# url = "http://localhost:8001/"
# response = requests.get(url, proxies={"http": None, "https": None})

# # url = "http://localhost:8001/"
# # response = requests.get(url, proxies={"http": None, "https": None})
# # print(response.status_code)
# # print(response.text)



# Run in the same terminal as your curl command
# ip netns identify $$
# Then run from your Python script




import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect(('0.0.0.0', 8123))
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
finally:
    s.close()