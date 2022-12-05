## A GPT-J Powered Character - Chatbot Template ##
The intent of this template is to serve as a quick intro guide for fellow developers 
looking to build and / or monetize GPT-J based character chatbots on Web and Native Android.

## Live Web Demo ##
![figure](images/img.png)

[Jeanie](https://messengerx.io/jeanie) is a GPT-J powered virtual girlfriend based on the above code

## Requirements for running it locally on laptop ##
* Windows / Mac / Linux with Git installed
* Python 3.5+
* MessengerX.io API Token - FREE for Indie Developers
* NLPCloud.io Token - (Pay as you Go plan - Maybe be required)
* Ngrok for Tunneling

## Get MessengerX.io API Key ##
* Get your FREE Developer API Token via [MessengerX.io](https://portal.messengerx.io) 
  or by [emailing us](mailto:connect@machaao.com) and replace it in the ```.env``` file under the entry ```API_TOKEN```
* MessengerX.io allows you to build and integrate a custom chatbot in your website or app


## Get NLPCloud.io API Key ##
* You can acquire an NLPCloud API Key via [NLP Cloud](https://nlpcloud.io) and replace it in the ```.env``` file under the entry
```NLP_CLOUD_TOKEN```
  
## Get Dashbot.io API KEY ##
* You can acquire the API Key via [Dashbot.io](https://dashbot.io) and replace it in the ```.env``` file under the entry
```DASHBOT_KEY```
  
## Local Setup ##
### Download or clone this repository ###
```
git clone git@github.com:machaao/gpt-j-chatbot.git

cd gpt-j-chatbot
```


### Install requirements ###
```bash
pip install -r requirements.txt
```

### Create a new .env file in the gpt-j-chatbot directory ###
```bash
nano -w .env
```

```
API_TOKEN = <Machaao API Token>
NLP_CLOUD_TOKEN = <NLP Cloud Token> 
BASE_URL = <Machaao Base Url> ex: https://ganglia-dev.machaao.com
NAME = <Your Bot Display Name> ex: Jess
DASHBOT_KEY = <Your dashbot token> (Optional) (Conversational Analytics)

# BOT PARAMS - Unset Parameters would use their default values.
# Don't use Top_p and Temperature parameters simultaneously.
# Default Values are mentioned below.
# The higher this value, the less deterministic the result will be
TOP_P="1.0" 
# The higher this value, the less deterministic the result will be
TEMPERATURE="0.8"
# The lower this value, the less likely GPT-J is going to generate off-topic text
TOP_K="50"
# The maximum number of tokens that the generated text should contain
MAX_LENGTH="50"
```
For better understanding of the above GPT-J parameters, check out the [nlpcloud.io](https://docs.nlpcloud.io/#generation) docs


### Modify logic/prompt.txt to change the character ###
```
This is a discussion between [user] and [name]
[name] is a very understanding girl.
```

### Modify the core() function in logic/bot_logic.py to personalize responses ###
```
def core(self, req: str, user_id: str):
```
* Refer to [platform documentation](https://messengerx.rtfd.iio) for personalization options

### Run the chatbot server from the root directory of the repo ###
```
python app.py
```

### Start ngrok.io tunnel ###
```
ngrok http 5000
```

### Update your webhook ###
Update your bot Webhook URL at [MessengerX.io Portal](https://portal.messengerx.io) with the url provided as shown below to continue development
```
Webhook Url: https://<TUNNEL-HOST>/webhooks/machaao/incoming
```
If you use [Ngrok.io](https://ngrok.io), your webhook URL would be of the format as in the example below
```
https://1234-115-187-40-104.ngrok.io/webhooks/machaao/incoming 
```
![figure](images/mx_screenshot.png)

### Test your bot:
Visit: ```https://dev.messengerx.io/<bot-name>```


## Remote Setup (Heroku) ##

We are assuming you have access to a [heroku account](https://heroku.com)
and have installed heroku command line client for your OS.

### Login to Heroku ###
```
heroku login
```

### Create a new app on Heroku and note down your heroku app name
```
heroku create
```

### Commit changes and push the repository to Heroku ###
```
git commit -m ".env updated"
git push heroku master
```

### Open the logs to confirm successful Deployment ###
```
heroku logs --tail
```

### Update your webhook ###
Update your bot Webhook URL at [MessengerX.io Portal](https://portal.messengerx.io) with the heroku app url
```
Webhook Url: <YOUR-HEROKU-APP-URL>/webhooks/machaao/incoming
```

### Test your bot:
Visit: ```https://dev.messengerx.io/<your-character-name>```


## Notes / Additional Resources ##
* Please note that this document isn't meant to be used as a guide for production environment setup.
* Please note to get a downloadable APK for your character, please contact us at [connect@machaao.com](mailto:connect@machaao.com)

[![Gitter](https://badges.gitter.im/messengerx-io/community.svg)](https://gitter.im/messengerx-io/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)  
