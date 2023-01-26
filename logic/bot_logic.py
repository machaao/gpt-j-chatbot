import base64
from datetime import datetime
import json
import nlpcloud
import nltk
import requests
from requests.structures import CaseInsensitiveDict
from dotenv import load_dotenv
import os

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, GPTNeoForCausalLM, GPTNeoXForCausalLM
import torch

load_dotenv()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = os.environ.get("MODEL_NAME", "")
MAX_HISTORY_LENGTH=os.environ.get("HISTORY_LENGTH", 5)
MODEL_X = os.environ.get("MODEL_X", False)

model = None

if MODEL_NAME:
    print(f"loading {MODEL_NAME} on local, on device {device}, please wait...")
    if MODEL_X==False:
        model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME).to(device)
    else:
        model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
else:
    print(f"no model name found - please check your .env file, gonna try to use nlpcloud.io")

ban_words = ["nigger", "negro", "nazi", "faggot", "murder", "suicide"]

# list of banned input words
c = 'UTF-8'


def send(url, headers, payload=None):
    if payload:
        print("sending post to platform: " + str(payload))
        response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
        print("response from the platform: " + str(response.text))
    else:
        response = requests.request("GET", url, headers=headers)

    return response


def get_details(api_token, base_url):
    _cache_ts_param = str(datetime.now().timestamp())
    e = "L3YxL2JvdHMvY29uZmlnP3Y9"
    check = base64.b64decode(e).decode(c)
    url = f"{base_url}{check}{_cache_ts_param}"
    headers = {
        "api_token": api_token,
        "Content-Type": "application/json",
    }

    response = send(url, headers)

    if response and response.status_code == 200:
        return response.json()
    else:
        return {}


class BotLogic:
    def __init__(self):
        # Initializing Config Variables
        self.api_token = os.environ.get("API_TOKEN")
        self.base_url = os.environ.get("BASE_URL", "https://ganglia.machaao.com")
        self.nlp_cloud_token = os.environ.get("NLP_CLOUD_TOKEN")
        self.name = os.environ.get("NAME")
        self.limit = os.environ.get("LIMIT", 'True')
        self.prefix = self.read_prompt(self.name)

        # Bot config
        self.top_p = os.environ.get("TOP_P", 1.0)
        self.top_k = os.environ.get("TOP_K", 20)
        self.temp = os.environ.get("TEMPERATURE", 0.3)
        self.max_length = os.environ.get("MAX_LENGTH", 50)
        self.validate_bot_params()

    # noinspection DuplicatedCode
    def validate_bot_params(self):
        print("Setting up Bot server with parameters:")
        if self.top_p is not None and self.temp is not None:
            print("Temperature and Top_p parameters can't be used together. Using default value of top_p")
            self.top_p = 1.0

        if self.temp is not None:
            self.temp = float(self.temp)
            if self.temp < 0.0 or self.temp > 1.0:
                raise Exception("Temperature parameter must be between 0 and 1")
        else:
            self.temp = 0.8
        print(f"Temperature = {self.temp}")

        if self.top_p is not None:
            self.top_p = float(self.top_p)
            if self.top_p < 0.0 or self.top_p > 1.0:
                raise Exception("Top_p parameter must be between 0 and 1")
        else:
            self.top_p = 1.0
        print(f"Top_p = {self.top_p}")

        if self.top_k is not None:
            self.top_k = int(self.top_k)
            if self.top_k > 1000:
                raise Exception("Top_k parameter must be less than 1000")
        else:
            self.top_k = 50
        print(f"Top_k = {self.top_k}")

        if self.max_length is not None:
            self.max_length = int(self.max_length)
            if self.max_length > 1024:
                raise Exception("Max_length parameter must be less than 1024")
        else:
            self.max_length = 50
        print(f"Max_Length = {self.max_length}")

    @staticmethod
    def read_prompt(name):
        file_name = "./logic/prompt.txt"
        with open(file_name) as f:
            prompt = f.read()

        return prompt.replace("name]", f"{name}]")

    def get_recent(self, user_id: str):
        count = MAX_HISTORY_LENGTH
        ## please don't edit the lines below
        e = "L3YxL2NvbnZlcnNhdGlvbnMvaGlzdG9yeS8="
        check = base64.b64decode(e).decode(c)
        url = f"{self.base_url}{check}{user_id}/{count}"

        headers = CaseInsensitiveDict()
        headers["api_token"] = self.api_token
        headers["Content-Type"] = "application/json"

        resp = requests.get(url, headers=headers)

        if resp.status_code == 200:
            return resp.json()

    @staticmethod
    def parse(data):
        msg_type = data.get('type')
        if msg_type == "outgoing":
            msg_data = json.loads(data['message'])
            msg_data_2 = json.loads(msg_data['message']['data']['message'])

            if msg_data_2 and msg_data_2.get('text', ''):
                text_data = msg_data_2['text']
            elif msg_data_2 and msg_data_2['attachment'] and msg_data_2['attachment'].get('payload', '') and \
                    msg_data_2['attachment']['payload'].get('text', ''):
                text_data = msg_data_2['attachment']['payload']['text']
            else:
                text_data = ""
        else:
            msg_data = json.loads(data['incoming'])
            if msg_data['message_data']['text']:
                text_data = msg_data['message_data']['text']
            else:
                text_data = ""

        return msg_type, text_data

    def core(self, req: str, label: str, user_id: str, client: str, sdk: str, action_type: str, api_token: str):
        print(
            "input text: " + req + ", label: " + label + ", user_id: " + user_id + ", client: " + client + ", sdk: " + sdk
            + ", action_type: " + action_type + ", api_token: " + api_token)

        bot = get_details(api_token, self.base_url)
        name = self.name
        _prompt = self.prefix

        if not bot:
            return False, "Oops, the chat bot doesn't exist or is not active at the moment"
        else:
            name = bot.get("displayName", name)

        if _prompt:
            _prompt = _prompt.replace("bot_name", f"{name}")

        valid = True

        # intents = ["default", "balance"]

        recent_text_data = []

        if str.lower(req.strip()) != "hi":
            recent_text_data = self.get_recent(user_id)  # [] -> blank for testing

        recent_convo_length = len(recent_text_data)

        print(f"len of returned history: {recent_convo_length}")

        # text = word_tokenize(req)
        # tags = nltk.pos_tag(text)
        # print(f"tags: {tags}")

        #

        history = str.strip(_prompt) + "\n###\n"

        banned = any(ele in req for ele in ban_words)

        if banned:
            print(f"banned input:" + str(req) + ", id: " + user_id)
            return False, "Oops, please refrain from such words"

        if recent_convo_length > 0:
            for text in recent_text_data[::-1]:
                msg_type, text_data = self.parse(text)
                if text_data and len(text_data) < 200 and "error" not in str.lower(text_data) and "oops" not in str.lower(text_data):
                    if msg_type is not None:
                        # outgoing msg - bot msg
                        history += f"{name}: " + text_data
                    else:
                        # incoming msg - user msg
                        history += "stranger: " + text_data

                    history += "\n"

                    if msg_type == "outgoing":
                        history += "###\n"
        else:
            history += f"stranger: " + str(req) + "\n"

        # history += f"{name}: \n"
        # Max input size = 2048 tokens
        try:
            print(f"processing prompt:\n{history}")
            if model:
                resp = self.process_via_local(req, name, history)
            else:
                resp = self.process_via_nlpcloud(name, history)

            # reply = str.capitalize(resp)
            # print(history + reply)
            return valid, resp
        except Exception as e:
            print(f"error - {e}, for {user_id}")
            return False, "Oops, I am feeling a little overwhelmed with messages\nPlease message me later"

    def process_via_nlpcloud(self, name, prompt):
        _client = nlpcloud.Client("gpt-j", self.nlp_cloud_token, gpu=True)
        generation = _client.generation(prompt,
                                        min_length=1,
                                        max_length=self.max_length,
                                        top_k=self.top_k,
                                        top_p=self.top_p,
                                        temperature=self.temp,
                                        length_no_input=True,
                                        end_sequence="\n###",
                                        remove_end_sequence=True,
                                        remove_input=True)
        resp = str.strip(generation["generated_text"])
        output = str.replace(resp, f"{name}:", "")
        return output

    def process_via_local(self, req, name, prompt):
        end_sequence = "###\n"

        input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
        sens = pad_sequence(input_ids, batch_first=True, padding_value=-1)
        attention_mask = (sens != -1).long()
        max_length = int(len(prompt) + len(req) + 10)

        if max_length > 2048:
            max_length = 2048

        gen_tokens = model.generate(
            sens,
            do_sample=True,
            temperature=0.9,
            top_p=self.top_p,
            top_k=self.top_k,
            max_length=max_length,
            attention_mask=attention_mask,
            eos_token_id=int(tokenizer.convert_tokens_to_ids(end_sequence))
        )

        gen = tokenizer.batch_decode(gen_tokens)
        gen_text = gen[0]

        gen_text = str.replace(str(gen_text), prompt, "")
        # print("returned: " + gen_text)

        gen_text = str.split(gen_text, "\n")
        first_match = None

        if len(gen_text) > 0:
            first_match = gen_text[0]

            for g in gen_text:
                if len(g) > 5 and f"{name}:" in g and g not in prompt:
                    gen_text = g
                    break

        if not gen_text:
            gen_text = first_match

        output = str.replace(gen_text, f"{name}:", "")

        print("output text: " + output)

        return str.strip(output)
