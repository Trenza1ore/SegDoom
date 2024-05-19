import os
import time
import requests
import traceback
from functools import partial

from base64 import b64encode

# load config
try:
    with open("webhook.config", 'r', encoding="utf-8") as f:
        WEBHOOK_URL = f.readline().strip()
        IMGBB_KEY = f.readline().strip()
    print("Successfully loaded webhook and imgbb api")
except:
    WEBHOOK_URL = "***REMOVED***"
    IMGBB_KEY = "***REMOVED***"
print(f"Loaded API:\nDiscord Webhook: {WEBHOOK_URL}\nImgBB: {IMGBB_KEY}")

if not os.path.exists("history"):
    os.mkdir("history")

def try_upload(f: partial, max_retry: int=3, wait_time: float=0.5, msg: str="Failed"):
    for _ in [0] * max_retry:
        try:
            return f()
        except:
            time.sleep(wait_time)
    raise Exception(msg)

class discord_bot:
    """
    A discord bot class that utilizes discord webhook to send stats and plots during training 
    """    
    def __init__(self, url: str = WEBHOOK_URL, extra: str=''):
        current_time = time.strftime("%y-%m-%d-%H-%M-%S")
        self.url = url
        self.last_upload = []
        self.data = {}
        self.epoch = -1
        self.path = "history/" + current_time
        os.mkdir(self.path)
        with open(self.path+'/'+extra, 'w') as f:
            f.write(current_time)
        self.send_string("**New Run: ** " + current_time + " " + extra)
    
    def update_data_img(self, epoch_num: int):
        content = f"Epoch {epoch_num+1}"
        self.data = {
            "content" : content,
            "username" : "Doom Guy",
            "embeds" : [{
                "image" : {"url" : img},
                "type" : "rich"
                } for img in self.last_upload]
        }

    def send_img(self, epoch_num: int):
        self.last_upload = []
        epoch_name = epoch_num + 1 if isinstance(epoch_num, int) else str(epoch_num)
        
        with open(f"{self.path}/current.png", "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {
                "key" : IMGBB_KEY,
                "image" : b64encode(file.read()),
                "name" : f"test{self.path}"
            }
            
        try:
            f = partial(requests.post, url, payload)
            result = try_upload(f, msg=f"Unable to upload images for epoch {epoch_name}")
            self.last_upload.append(result.json()["data"]["display_url"])
        except:
            print(f"Unable to upload images for epoch {epoch_name}")
            return
            
        try:
            self.update_data_img(epoch_name)
            f = partial(requests.post, self.url, json=self.data)
            result = try_upload(f, msg=f"Unable to send images for epoch {epoch_name}")
        except:
            print(f"Unable to send images for epoch {epoch_name}")
    
    def send_string(self, content: str):
        stat_data = {
            "content" : content,
            "username" : "Doom Guy"
        }
        try:
            requests.post(self.url, json = stat_data)
        except:
            print(f"Unable to send stats")
    
    def send_error(self, e: Exception):
        err_msg = ''.join(traceback.format_exception(e))
        self.send_string(err_msg)
        print(err_msg)