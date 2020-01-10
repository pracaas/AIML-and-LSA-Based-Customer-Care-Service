#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:36:41 2018

@author: prakash
"""

#!/usr/bin/python3
import os
import aiml
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import pyttsx3
#import os
bot = ChatBot('Bot')


bot.set_trainer(ListTrainer)


import speech_recognition as sr

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio= r.adjust_for_ambient_noise(source)
        #print(r.energy_threshold,"threshold level")
        print("...")
        audio = r.listen(source)
    try:
        aname = r.recognize_google(audio)
        #print("test on listen")
        return aname
    except sr.UnknownValueError:
        #print("I cannot hear you sir")
        #speake("I cannot hear you sir")
        
        return "I cannot hear you sir"
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def speake(string):
    engine = pyttsx3.init()
    volume =engine.getProperty('volume')
    engine.setProperty('volume',volume)
    engine.say(string)
    engine.runAndWait()
    engine.setProperty('rate',228)
    

k = aiml.Kernel()
def trainbot():
    BRAIN_FILE="brain.dump"
    
    
    
    # To increase the startup speed of the bot it is
    # possible to save the parsed aiml files as a
    # dump. This code checks if a dump exists and
    # otherwise loads the aiml from the xml files
    # and saves the brain dump.
    if os.path.exists(BRAIN_FILE):
        print("Loading from brain file: " + BRAIN_FILE)
        k.loadBrain(BRAIN_FILE)
    else:
        print("Parsing aiml files")
        k.bootstrap(learnFiles="std-startup.aiml", commands="load aiml b")
        print("Saving brain file: " + BRAIN_FILE)
        k.saveBrain(BRAIN_FILE)

# Endless loop which passes the input to the bot and prints
# its response
    
#speake("Please Speake")
#print("Please speak:")
    
def input_response(typ,txt):
    if "audio" in typ:
        message = listen()
        return k.respond(message)
    else :
        message = k.respond(txt)
        return message

