#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 21:58:09 2018

@author: prakash
"""

from  AppKit import NSSpeechSynthesizer

def speakloud(speech):
    nssp = NSSpeechSynthesizer
    ve = nssp.alloc().init()
    
    ve.setVoice_("com.apple.speech.synthesis.voice.samantha")
    ve.startSpeakingString_(speech)

    #for voice in nssp.availableVoices():
    #   print(voice)
#speakloud("One love for the mother cried")
