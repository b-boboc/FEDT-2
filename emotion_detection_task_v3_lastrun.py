#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.4),
    on January 19, 2025, at 19:38
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.4'
expName = 'emotion_detection_task_v3'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [960, 540]
_loggingLevel = logging.getLevel('exp')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Andrew\\My Drive\\fiverr 2024\\biancaboboc484\\moodhormonesfaceemotions_modified\\emotion_detection_task_v3_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('intro_key') is None:
        # initialise intro_key
        intro_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_key',
        )
    if deviceManager.getDevice('training_instructions_key') is None:
        # initialise training_instructions_key
        training_instructions_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='training_instructions_key',
        )
    if deviceManager.getDevice('training_trial_key') is None:
        # initialise training_trial_key
        training_trial_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='training_trial_key',
        )
    if deviceManager.getDevice('training_trial_2_key') is None:
        # initialise training_trial_2_key
        training_trial_2_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='training_trial_2_key',
        )
    if deviceManager.getDevice('instructions_key') is None:
        # initialise instructions_key
        instructions_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instructions_key',
        )
    if deviceManager.getDevice('practice_first_step_valid_keys') is None:
        # initialise practice_first_step_valid_keys
        practice_first_step_valid_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='practice_first_step_valid_keys',
        )
    if deviceManager.getDevice('practice_first_step_invalid_keys') is None:
        # initialise practice_first_step_invalid_keys
        practice_first_step_invalid_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='practice_first_step_invalid_keys',
        )
    if deviceManager.getDevice('practice_trial_valid_keys') is None:
        # initialise practice_trial_valid_keys
        practice_trial_valid_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='practice_trial_valid_keys',
        )
    if deviceManager.getDevice('practice_trial_invalid_keys') is None:
        # initialise practice_trial_invalid_keys
        practice_trial_invalid_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='practice_trial_invalid_keys',
        )
    if deviceManager.getDevice('first_morph_step_valid_keys') is None:
        # initialise first_morph_step_valid_keys
        first_morph_step_valid_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='first_morph_step_valid_keys',
        )
    if deviceManager.getDevice('first_morph_step_invalid_keys') is None:
        # initialise first_morph_step_invalid_keys
        first_morph_step_invalid_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='first_morph_step_invalid_keys',
        )
    if deviceManager.getDevice('rating_valid_keys') is None:
        # initialise rating_valid_keys
        rating_valid_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rating_valid_keys',
        )
    if deviceManager.getDevice('rating_invalid_keys') is None:
        # initialise rating_invalid_keys
        rating_invalid_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rating_invalid_keys',
        )
    if deviceManager.getDevice('end_key') is None:
        # initialise end_key
        end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_key',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "setup" ---
    # Run 'Begin Experiment' code from setup_code
    # Variables, durations/sizes etc.
    # Any variables used across multiple routines are specified here
    # Variables only used within one routine are specified in their own code component
    fixationDur = 1
    
    instruction_size = 0.03
    
    cue_size = 0.05
    
    # Position for stimuli
    imagePos = [0, 0.05]
    # Position for keypress mapping image
    keypressPos = [0, -0.35]
    # Position for invalid keypress text
    keypressMsgPos = [0, 0.4]
    
    # --- Initialize components for Routine "intro" ---
    # Run 'Begin Experiment' code from intro_code
    introText = "Participant ID:" + str(expInfo['participant']) + "\n\n Please answer the following questions:\n\n Press ENTER to continue once you have responded to each question. \n\n"
    questionOne = "On what DAY were you born (e.g., July 16, 1985 - 16 is the DAY):"
    questionTwo = "What are the FIRST THREE letters of you mother’s FIRST name?"
    questionThree = "What is the FIRST letter of your middle name? (if you do not have a middle name please put the letter ‘X’)"
    
    intro_leftPos = -0.3
    intro_rightPos = 0.3
    intro_topPos = 0.25
    intro_midPos = 0
    intro_bottomPos = -0.25
    intro_wrapWidth = 0.4
    intro_text = visual.TextStim(win=win, name='intro_text',
        text=introText,
        font='Open Sans',
        pos=(0, 0.4), height=instruction_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    intro_q1 = visual.TextStim(win=win, name='intro_q1',
        text=questionOne,
        font='Open Sans',
        pos=(intro_leftPos, intro_topPos), height=instruction_size, wrapWidth=intro_wrapWidth, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    intro_q2 = visual.TextStim(win=win, name='intro_q2',
        text=questionTwo,
        font='Open Sans',
        pos=(intro_leftPos, intro_midPos), height=instruction_size, wrapWidth=intro_wrapWidth, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_q3 = visual.TextStim(win=win, name='intro_q3',
        text=questionThree,
        font='Open Sans',
        pos=(intro_leftPos, intro_bottomPos), height=instruction_size, wrapWidth=intro_wrapWidth, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    intro_box_1 = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Open Sans',
         pos=(intro_rightPos, intro_topPos),     letterHeight=0.05,
         size=(0.1, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.02, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='intro_box_1',
         depth=-5, autoLog=True,
    )
    intro_box_2 = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Open Sans',
         pos=(intro_rightPos, intro_midPos),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.02, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='intro_box_2',
         depth=-6, autoLog=True,
    )
    intro_box_3 = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Open Sans',
         pos=(intro_rightPos, intro_bottomPos),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.02, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='intro_box_3',
         depth=-7, autoLog=True,
    )
    intro_key = keyboard.Keyboard(deviceName='intro_key')
    intro_mouse = event.Mouse(win=win)
    x, y = [None, None]
    intro_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "training_instructions" ---
    training_instructions_text = visual.TextStim(win=win, name='training_instructions_text',
        text='You will be completing a facial emotion detection task. You will be asked to use your keyboard to indicate what emotion you see. \n\nThere will be a training session before the main task. You will be asked to identify the emotions within 7 faces, and will be told when your response is wrong. Then you will be asked to identify the emotions of 12 more faces, without any feedback. For each face, try to respond as quickly as possible. \n\nPress the SPACE BAR to progress to the training session.',
        font='Open Sans',
        pos=(0, 0), height=instruction_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    training_instructions_key = keyboard.Keyboard(deviceName='training_instructions_key')
    
    # --- Initialize components for Routine "training_message" ---
    training_message_text = visual.TextStim(win=win, name='training_message_text',
        text='Training Session',
        font='Open Sans',
        pos=(0, 0), height=instruction_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "isi" ---
    isi_text = visual.TextStim(win=win, name='isi_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "training_trial" ---
    # Run 'Begin Experiment' code from training_trial_code
    trainingMsg = False
    training_trial_text = visual.TextStim(win=win, name='training_trial_text',
        text='For each image indicate what emotion you see using the keyboard mapping shown below.',
        font='Open Sans',
        pos=keypressMsgPos, height=cue_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    training_trial_incorrect_message = visual.TextStim(win=win, name='training_trial_incorrect_message',
        text='incorrect!\ntry again.',
        font='Open Sans',
        pos=(0.45, 0), height=cue_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    training_trial_image = visual.ImageStim(
        win=win,
        name='training_trial_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=imagePos, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    training_trial_key_mapping = visual.ImageStim(
        win=win,
        name='training_trial_key_mapping', 
        image='key_mapping.png', mask=None, anchor='center',
        ori=0.0, pos=keypressPos, size=(0.5148, 0.264),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    training_trial_key = keyboard.Keyboard(deviceName='training_trial_key')
    
    # --- Initialize components for Routine "isi" ---
    isi_text = visual.TextStim(win=win, name='isi_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "training_trial_2" ---
    training_trial_2_text = visual.TextStim(win=win, name='training_trial_2_text',
        text='',
        font='Open Sans',
        pos=keypressMsgPos, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    training_trial_2_image = visual.ImageStim(
        win=win,
        name='training_trial_2_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=imagePos, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    training_trial_2_key_mapping = visual.ImageStim(
        win=win,
        name='training_trial_2_key_mapping', 
        image='key_mapping.png', mask=None, anchor='center',
        ori=0.0, pos=keypressPos, size=(0.5148, 0.264),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    training_trial_2_key = keyboard.Keyboard(deviceName='training_trial_2_key')
    
    # --- Initialize components for Routine "instructions" ---
    instructions_text = visual.TextStim(win=win, name='instructions_text',
        text='Main Task \n\nYou will be shown an image of a neutral face which, over the span of 15 images, will gradually morph/change into one of the following emotions: disgust, fear, sad, angry, happy, or surprise. For every image in the morph indicate what emotion you see. We are interested in how fast and accurately you can identify the emotion. \n \nThe first face of each morph will appear neutral. Begin by pressing the SPACE BAR to indicate this. Do this for all of the following faces that appear neutral. Once you see an emotion in the face, press the key corresponding to the emotion that you see. If you see an emotion but are unsure what it is, please only guess the emotion you see when you are reasonably confident. For each image try to respond as quickly as possible. \n \nThere will be a practice trial, then 24 experimental trials. \n\nPress SPACE BAR to begin the practice trial.',
        font='Open Sans',
        pos=(0, 0), height=instruction_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instructions_key = keyboard.Keyboard(deviceName='instructions_key')
    
    # --- Initialize components for Routine "practice_message" ---
    practice_message_text = visual.TextStim(win=win, name='practice_message_text',
        text='Practice Block',
        font='Open Sans',
        pos=(0, 0), height=instruction_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "practice_first_step" ---
    practice_first_step_text = visual.TextStim(win=win, name='practice_first_step_text',
        text='',
        font='Open Sans',
        pos=keypressMsgPos, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    practice_first_step_image = visual.ImageStim(
        win=win,
        name='practice_first_step_image', 
        image='stimuli/practice/PT_00.jpg', mask=None, anchor='center',
        ori=0.0, pos=imagePos, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    practice_first_step_key_mapping = visual.ImageStim(
        win=win,
        name='practice_first_step_key_mapping', 
        image='key_mapping_first.png', mask=None, anchor='center',
        ori=0.0, pos=keypressPos, size=(0.5148, 0.264),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    practice_first_step_valid_keys = keyboard.Keyboard(deviceName='practice_first_step_valid_keys')
    practice_first_step_invalid_keys = keyboard.Keyboard(deviceName='practice_first_step_invalid_keys')
    
    # --- Initialize components for Routine "isi" ---
    isi_text = visual.TextStim(win=win, name='isi_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "practice_trial" ---
    practice_trial_text = visual.TextStim(win=win, name='practice_trial_text',
        text='',
        font='Open Sans',
        pos=keypressMsgPos, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    practice_trial_key_mapping = visual.ImageStim(
        win=win,
        name='practice_trial_key_mapping', 
        image='key_mapping.png', mask=None, anchor='center',
        ori=0.0, pos=keypressPos, size=(0.5148, 0.264),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    practice_trial_image = visual.ImageStim(
        win=win,
        name='practice_trial_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=imagePos, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    practice_trial_valid_keys = keyboard.Keyboard(deviceName='practice_trial_valid_keys')
    practice_trial_invalid_keys = keyboard.Keyboard(deviceName='practice_trial_invalid_keys')
    
    # --- Initialize components for Routine "main_message" ---
    main_message_text = visual.TextStim(win=win, name='main_message_text',
        text='Main Experimental Block',
        font='Open Sans',
        pos=(0, 0), height=instruction_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "fixation" ---
    fixation_text = visual.TextStim(win=win, name='fixation_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "is_first_step" ---
    
    # --- Initialize components for Routine "isi" ---
    isi_text = visual.TextStim(win=win, name='isi_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "first_morph_step" ---
    first_morph_step_text = visual.TextStim(win=win, name='first_morph_step_text',
        text='',
        font='Open Sans',
        pos=keypressMsgPos, height=cue_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    first_morph_step_image = visual.ImageStim(
        win=win,
        name='first_morph_step_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=imagePos, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    first_morph_step_key_mapping = visual.ImageStim(
        win=win,
        name='first_morph_step_key_mapping', 
        image='key_mapping_first.png', mask=None, anchor='center',
        ori=0.0, pos=keypressPos, size=(0.5148, 0.264),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    first_morph_step_valid_keys = keyboard.Keyboard(deviceName='first_morph_step_valid_keys')
    first_morph_step_invalid_keys = keyboard.Keyboard(deviceName='first_morph_step_invalid_keys')
    
    # --- Initialize components for Routine "add_data" ---
    
    # --- Initialize components for Routine "isnt_first_step" ---
    
    # --- Initialize components for Routine "isi" ---
    isi_text = visual.TextStim(win=win, name='isi_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rating" ---
    rating_text = visual.TextStim(win=win, name='rating_text',
        text='',
        font='Open Sans',
        pos=keypressMsgPos, height=cue_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    rating_image = visual.ImageStim(
        win=win,
        name='rating_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=imagePos, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    rating_key_mapping = visual.ImageStim(
        win=win,
        name='rating_key_mapping', 
        image='key_mapping.png', mask=None, anchor='center',
        ori=0.0, pos=keypressPos, size=(0.5148, 0.264),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    rating_valid_keys = keyboard.Keyboard(deviceName='rating_valid_keys')
    rating_invalid_keys = keyboard.Keyboard(deviceName='rating_invalid_keys')
    
    # --- Initialize components for Routine "add_data" ---
    
    # --- Initialize components for Routine "end" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text='Thank you for your participation. \n\nREMEMBER THIS CODE: 955\nYou will need to enter this code in Survey Monkey.\n\n\nPlease return to Survey Monkey to finish the experiment. \n\nPress ENTER to end this task. ',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_key = keyboard.Keyboard(deviceName='end_key')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "setup" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('setup.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    setupComponents = []
    for thisComponent in setupComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "setup" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in setupComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "setup" ---
    for thisComponent in setupComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('setup.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "setup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro.started', globalClock.getTime(format='float'))
    intro_box_1.reset()
    intro_box_2.reset()
    intro_box_3.reset()
    intro_key.keys = []
    intro_key.rt = []
    _intro_key_allKeys = []
    # setup some python lists for storing info about the intro_mouse
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    introComponents = [intro_text, intro_q1, intro_q2, intro_q3, intro_box_1, intro_box_2, intro_box_3, intro_key, intro_mouse]
    for thisComponent in introComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from intro_code
        if intro_box_1.text != "":
            if intro_box_2.text != "":
                if intro_box_3.text != "":
                    responded = True
        else:
            responded = False
        
        # *intro_text* updates
        
        # if intro_text is starting this frame...
        if intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text.frameNStart = frameN  # exact frame index
            intro_text.tStart = t  # local t and not account for scr refresh
            intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_text.started')
            # update status
            intro_text.status = STARTED
            intro_text.setAutoDraw(True)
        
        # if intro_text is active this frame...
        if intro_text.status == STARTED:
            # update params
            pass
        
        # *intro_q1* updates
        
        # if intro_q1 is starting this frame...
        if intro_q1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_q1.frameNStart = frameN  # exact frame index
            intro_q1.tStart = t  # local t and not account for scr refresh
            intro_q1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_q1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_q1.started')
            # update status
            intro_q1.status = STARTED
            intro_q1.setAutoDraw(True)
        
        # if intro_q1 is active this frame...
        if intro_q1.status == STARTED:
            # update params
            pass
        
        # *intro_q2* updates
        
        # if intro_q2 is starting this frame...
        if intro_q2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_q2.frameNStart = frameN  # exact frame index
            intro_q2.tStart = t  # local t and not account for scr refresh
            intro_q2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_q2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_q2.started')
            # update status
            intro_q2.status = STARTED
            intro_q2.setAutoDraw(True)
        
        # if intro_q2 is active this frame...
        if intro_q2.status == STARTED:
            # update params
            pass
        
        # *intro_q3* updates
        
        # if intro_q3 is starting this frame...
        if intro_q3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_q3.frameNStart = frameN  # exact frame index
            intro_q3.tStart = t  # local t and not account for scr refresh
            intro_q3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_q3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_q3.started')
            # update status
            intro_q3.status = STARTED
            intro_q3.setAutoDraw(True)
        
        # if intro_q3 is active this frame...
        if intro_q3.status == STARTED:
            # update params
            pass
        
        # *intro_box_1* updates
        
        # if intro_box_1 is starting this frame...
        if intro_box_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_box_1.frameNStart = frameN  # exact frame index
            intro_box_1.tStart = t  # local t and not account for scr refresh
            intro_box_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_box_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_box_1.started')
            # update status
            intro_box_1.status = STARTED
            intro_box_1.setAutoDraw(True)
        
        # if intro_box_1 is active this frame...
        if intro_box_1.status == STARTED:
            # update params
            pass
        
        # *intro_box_2* updates
        
        # if intro_box_2 is starting this frame...
        if intro_box_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_box_2.frameNStart = frameN  # exact frame index
            intro_box_2.tStart = t  # local t and not account for scr refresh
            intro_box_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_box_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_box_2.started')
            # update status
            intro_box_2.status = STARTED
            intro_box_2.setAutoDraw(True)
        
        # if intro_box_2 is active this frame...
        if intro_box_2.status == STARTED:
            # update params
            pass
        
        # *intro_box_3* updates
        
        # if intro_box_3 is starting this frame...
        if intro_box_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_box_3.frameNStart = frameN  # exact frame index
            intro_box_3.tStart = t  # local t and not account for scr refresh
            intro_box_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_box_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_box_3.started')
            # update status
            intro_box_3.status = STARTED
            intro_box_3.setAutoDraw(True)
        
        # if intro_box_3 is active this frame...
        if intro_box_3.status == STARTED:
            # update params
            pass
        
        # *intro_key* updates
        waitOnFlip = False
        
        # if intro_key is starting this frame...
        if intro_key.status == NOT_STARTED and responded:
            # keep track of start time/frame for later
            intro_key.frameNStart = frameN  # exact frame index
            intro_key.tStart = t  # local t and not account for scr refresh
            intro_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_key.started')
            # update status
            intro_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_key.status == STARTED and not waitOnFlip:
            theseKeys = intro_key.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _intro_key_allKeys.extend(theseKeys)
            if len(_intro_key_allKeys):
                intro_key.keys = _intro_key_allKeys[-1].name  # just the last key pressed
                intro_key.rt = _intro_key_allKeys[-1].rt
                intro_key.duration = _intro_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # *intro_mouse* updates
        
        # if intro_mouse is starting this frame...
        if intro_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_mouse.frameNStart = frameN  # exact frame index
            intro_mouse.tStart = t  # local t and not account for scr refresh
            intro_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('intro_mouse.started', t)
            # update status
            intro_mouse.status = STARTED
            intro_mouse.mouseClock.reset()
            prevButtonState = intro_mouse.getPressed()  # if button is down already this ISN'T a new click
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in introComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from intro_code
    win.mouseVisible = False
    thisExp.addData('intro_box_1.text',intro_box_1.text)
    thisExp.addData('intro_box_2.text',intro_box_2.text)
    thisExp.addData('intro_box_3.text',intro_box_3.text)
    # check responses
    if intro_key.keys in ['', [], None]:  # No response was made
        intro_key.keys = None
    thisExp.addData('intro_key.keys',intro_key.keys)
    if intro_key.keys != None:  # we had a response
        thisExp.addData('intro_key.rt', intro_key.rt)
        thisExp.addData('intro_key.duration', intro_key.duration)
    # store data for thisExp (ExperimentHandler)
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "training_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('training_instructions.started', globalClock.getTime(format='float'))
    training_instructions_key.keys = []
    training_instructions_key.rt = []
    _training_instructions_key_allKeys = []
    # keep track of which components have finished
    training_instructionsComponents = [training_instructions_text, training_instructions_key]
    for thisComponent in training_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "training_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *training_instructions_text* updates
        
        # if training_instructions_text is starting this frame...
        if training_instructions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            training_instructions_text.frameNStart = frameN  # exact frame index
            training_instructions_text.tStart = t  # local t and not account for scr refresh
            training_instructions_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(training_instructions_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'training_instructions_text.started')
            # update status
            training_instructions_text.status = STARTED
            training_instructions_text.setAutoDraw(True)
        
        # if training_instructions_text is active this frame...
        if training_instructions_text.status == STARTED:
            # update params
            pass
        
        # *training_instructions_key* updates
        waitOnFlip = False
        
        # if training_instructions_key is starting this frame...
        if training_instructions_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            training_instructions_key.frameNStart = frameN  # exact frame index
            training_instructions_key.tStart = t  # local t and not account for scr refresh
            training_instructions_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(training_instructions_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'training_instructions_key.started')
            # update status
            training_instructions_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(training_instructions_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(training_instructions_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if training_instructions_key.status == STARTED and not waitOnFlip:
            theseKeys = training_instructions_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _training_instructions_key_allKeys.extend(theseKeys)
            if len(_training_instructions_key_allKeys):
                training_instructions_key.keys = _training_instructions_key_allKeys[-1].name  # just the last key pressed
                training_instructions_key.rt = _training_instructions_key_allKeys[-1].rt
                training_instructions_key.duration = _training_instructions_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in training_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "training_instructions" ---
    for thisComponent in training_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('training_instructions.stopped', globalClock.getTime(format='float'))
    # check responses
    if training_instructions_key.keys in ['', [], None]:  # No response was made
        training_instructions_key.keys = None
    thisExp.addData('training_instructions_key.keys',training_instructions_key.keys)
    if training_instructions_key.keys != None:  # we had a response
        thisExp.addData('training_instructions_key.rt', training_instructions_key.rt)
        thisExp.addData('training_instructions_key.duration', training_instructions_key.duration)
    thisExp.nextEntry()
    # the Routine "training_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "training_message" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('training_message.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    training_messageComponents = [training_message_text]
    for thisComponent in training_messageComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "training_message" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *training_message_text* updates
        
        # if training_message_text is starting this frame...
        if training_message_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            training_message_text.frameNStart = frameN  # exact frame index
            training_message_text.tStart = t  # local t and not account for scr refresh
            training_message_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(training_message_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'training_message_text.started')
            # update status
            training_message_text.status = STARTED
            training_message_text.setAutoDraw(True)
        
        # if training_message_text is active this frame...
        if training_message_text.status == STARTED:
            # update params
            pass
        
        # if training_message_text is stopping this frame...
        if training_message_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > training_message_text.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                training_message_text.tStop = t  # not accounting for scr refresh
                training_message_text.tStopRefresh = tThisFlipGlobal  # on global time
                training_message_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_message_text.stopped')
                # update status
                training_message_text.status = FINISHED
                training_message_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in training_messageComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "training_message" ---
    for thisComponent in training_messageComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('training_message.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    training_loop = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions/training_sequential.xlsx'),
        seed=None, name='training_loop')
    thisExp.addLoop(training_loop)  # add the loop to the experiment
    thisTraining_loop = training_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTraining_loop.rgb)
    if thisTraining_loop != None:
        for paramName in thisTraining_loop:
            globals()[paramName] = thisTraining_loop[paramName]
    
    for thisTraining_loop in training_loop:
        currentLoop = training_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTraining_loop.rgb)
        if thisTraining_loop != None:
            for paramName in thisTraining_loop:
                globals()[paramName] = thisTraining_loop[paramName]
        
        # --- Prepare to start Routine "isi" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('isi.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        isiComponents = [isi_text]
        for thisComponent in isiComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "isi" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.1:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *isi_text* updates
            
            # if isi_text is starting this frame...
            if isi_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                isi_text.frameNStart = frameN  # exact frame index
                isi_text.tStart = t  # local t and not account for scr refresh
                isi_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(isi_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'isi_text.started')
                # update status
                isi_text.status = STARTED
                isi_text.setAutoDraw(True)
            
            # if isi_text is active this frame...
            if isi_text.status == STARTED:
                # update params
                pass
            
            # if isi_text is stopping this frame...
            if isi_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > isi_text.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    isi_text.tStop = t  # not accounting for scr refresh
                    isi_text.tStopRefresh = tThisFlipGlobal  # on global time
                    isi_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'isi_text.stopped')
                    # update status
                    isi_text.status = FINISHED
                    isi_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in isiComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "isi" ---
        for thisComponent in isiComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('isi.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.100000)
        
        # --- Prepare to start Routine "training_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('training_trial.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from training_trial_code
        event.clearEvents()
        training_trial_image.setImage(trainingStim)
        training_trial_key.keys = []
        training_trial_key.rt = []
        _training_trial_key_allKeys = []
        # keep track of which components have finished
        training_trialComponents = [training_trial_text, training_trial_incorrect_message, training_trial_image, training_trial_key_mapping, training_trial_key]
        for thisComponent in training_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "training_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from training_trial_code
            trainingKeys = event.getKeys(keyList = ['d','f','v','space','n','j','k'])
            if len(trainingKeys) != 0:
                if trainingKeys[0] == trainingAns:
                    continueRoutine = False
                else:
                    trainingMsg = True
                    event.clearEvents()
            
            # *training_trial_text* updates
            
            # if training_trial_text is starting this frame...
            if training_trial_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                training_trial_text.frameNStart = frameN  # exact frame index
                training_trial_text.tStart = t  # local t and not account for scr refresh
                training_trial_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_trial_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_trial_text.started')
                # update status
                training_trial_text.status = STARTED
                training_trial_text.setAutoDraw(True)
            
            # if training_trial_text is active this frame...
            if training_trial_text.status == STARTED:
                # update params
                pass
            
            # *training_trial_incorrect_message* updates
            
            # if training_trial_incorrect_message is starting this frame...
            if training_trial_incorrect_message.status == NOT_STARTED and trainingMsg:
                # keep track of start time/frame for later
                training_trial_incorrect_message.frameNStart = frameN  # exact frame index
                training_trial_incorrect_message.tStart = t  # local t and not account for scr refresh
                training_trial_incorrect_message.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_trial_incorrect_message, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_trial_incorrect_message.started')
                # update status
                training_trial_incorrect_message.status = STARTED
                training_trial_incorrect_message.setAutoDraw(True)
            
            # if training_trial_incorrect_message is active this frame...
            if training_trial_incorrect_message.status == STARTED:
                # update params
                pass
            
            # *training_trial_image* updates
            
            # if training_trial_image is starting this frame...
            if training_trial_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                training_trial_image.frameNStart = frameN  # exact frame index
                training_trial_image.tStart = t  # local t and not account for scr refresh
                training_trial_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_trial_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_trial_image.started')
                # update status
                training_trial_image.status = STARTED
                training_trial_image.setAutoDraw(True)
            
            # if training_trial_image is active this frame...
            if training_trial_image.status == STARTED:
                # update params
                pass
            
            # *training_trial_key_mapping* updates
            
            # if training_trial_key_mapping is starting this frame...
            if training_trial_key_mapping.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                training_trial_key_mapping.frameNStart = frameN  # exact frame index
                training_trial_key_mapping.tStart = t  # local t and not account for scr refresh
                training_trial_key_mapping.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_trial_key_mapping, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_trial_key_mapping.started')
                # update status
                training_trial_key_mapping.status = STARTED
                training_trial_key_mapping.setAutoDraw(True)
            
            # if training_trial_key_mapping is active this frame...
            if training_trial_key_mapping.status == STARTED:
                # update params
                pass
            
            # *training_trial_key* updates
            waitOnFlip = False
            
            # if training_trial_key is starting this frame...
            if training_trial_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                training_trial_key.frameNStart = frameN  # exact frame index
                training_trial_key.tStart = t  # local t and not account for scr refresh
                training_trial_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_trial_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_trial_key.started')
                # update status
                training_trial_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(training_trial_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(training_trial_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if training_trial_key.status == STARTED and not waitOnFlip:
                theseKeys = training_trial_key.getKeys(keyList=['d', 'f', 'v', 'space', 'n', 'j', 'k'], ignoreKeys=["escape"], waitRelease=False)
                _training_trial_key_allKeys.extend(theseKeys)
                if len(_training_trial_key_allKeys):
                    training_trial_key.keys = _training_trial_key_allKeys[-1].name  # just the last key pressed
                    training_trial_key.rt = _training_trial_key_allKeys[-1].rt
                    training_trial_key.duration = _training_trial_key_allKeys[-1].duration
                    # was this correct?
                    if (training_trial_key.keys == str('trainingAns')) or (training_trial_key.keys == 'trainingAns'):
                        training_trial_key.corr = 1
                    else:
                        training_trial_key.corr = 0
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in training_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "training_trial" ---
        for thisComponent in training_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('training_trial.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from training_trial_code
        trainingMsg = False
        # check responses
        if training_trial_key.keys in ['', [], None]:  # No response was made
            training_trial_key.keys = None
            # was no response the correct answer?!
            if str('trainingAns').lower() == 'none':
               training_trial_key.corr = 1;  # correct non-response
            else:
               training_trial_key.corr = 0;  # failed to respond (incorrectly)
        # store data for training_loop (TrialHandler)
        training_loop.addData('training_trial_key.keys',training_trial_key.keys)
        training_loop.addData('training_trial_key.corr', training_trial_key.corr)
        if training_trial_key.keys != None:  # we had a response
            training_loop.addData('training_trial_key.rt', training_trial_key.rt)
            training_loop.addData('training_trial_key.duration', training_trial_key.duration)
        # the Routine "training_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'training_loop'
    
    
    # set up handler to look after randomisation of conditions etc
    training_loop_2 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions/training_random.xlsx'),
        seed=None, name='training_loop_2')
    thisExp.addLoop(training_loop_2)  # add the loop to the experiment
    thisTraining_loop_2 = training_loop_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTraining_loop_2.rgb)
    if thisTraining_loop_2 != None:
        for paramName in thisTraining_loop_2:
            globals()[paramName] = thisTraining_loop_2[paramName]
    
    for thisTraining_loop_2 in training_loop_2:
        currentLoop = training_loop_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTraining_loop_2.rgb)
        if thisTraining_loop_2 != None:
            for paramName in thisTraining_loop_2:
                globals()[paramName] = thisTraining_loop_2[paramName]
        
        # --- Prepare to start Routine "isi" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('isi.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        isiComponents = [isi_text]
        for thisComponent in isiComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "isi" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.1:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *isi_text* updates
            
            # if isi_text is starting this frame...
            if isi_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                isi_text.frameNStart = frameN  # exact frame index
                isi_text.tStart = t  # local t and not account for scr refresh
                isi_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(isi_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'isi_text.started')
                # update status
                isi_text.status = STARTED
                isi_text.setAutoDraw(True)
            
            # if isi_text is active this frame...
            if isi_text.status == STARTED:
                # update params
                pass
            
            # if isi_text is stopping this frame...
            if isi_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > isi_text.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    isi_text.tStop = t  # not accounting for scr refresh
                    isi_text.tStopRefresh = tThisFlipGlobal  # on global time
                    isi_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'isi_text.stopped')
                    # update status
                    isi_text.status = FINISHED
                    isi_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in isiComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "isi" ---
        for thisComponent in isiComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('isi.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.100000)
        
        # --- Prepare to start Routine "training_trial_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('training_trial_2.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from training_trial_2_code
        # Message that displays for invalid keypresses, blank to begin with
        keypressMsg = ""
        
        event.clearEvents()
        training_trial_2_image.setImage(training2Stim)
        training_trial_2_key.keys = []
        training_trial_2_key.rt = []
        _training_trial_2_key_allKeys = []
        # keep track of which components have finished
        training_trial_2Components = [training_trial_2_text, training_trial_2_image, training_trial_2_key_mapping, training_trial_2_key]
        for thisComponent in training_trial_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "training_trial_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from training_trial_2_code
            invalid = event.getKeys(keyList = ['a', 'b', 'c', 'e', 'g', 'h', 'i', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z'])
            
            if len(invalid) != 0:
                keypressMsg = "invalid keypress, please only use the keys indicated"
            
            # *training_trial_2_text* updates
            
            # if training_trial_2_text is starting this frame...
            if training_trial_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                training_trial_2_text.frameNStart = frameN  # exact frame index
                training_trial_2_text.tStart = t  # local t and not account for scr refresh
                training_trial_2_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_trial_2_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_trial_2_text.started')
                # update status
                training_trial_2_text.status = STARTED
                training_trial_2_text.setAutoDraw(True)
            
            # if training_trial_2_text is active this frame...
            if training_trial_2_text.status == STARTED:
                # update params
                training_trial_2_text.setText(keypressMsg, log=False)
            
            # *training_trial_2_image* updates
            
            # if training_trial_2_image is starting this frame...
            if training_trial_2_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                training_trial_2_image.frameNStart = frameN  # exact frame index
                training_trial_2_image.tStart = t  # local t and not account for scr refresh
                training_trial_2_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_trial_2_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_trial_2_image.started')
                # update status
                training_trial_2_image.status = STARTED
                training_trial_2_image.setAutoDraw(True)
            
            # if training_trial_2_image is active this frame...
            if training_trial_2_image.status == STARTED:
                # update params
                pass
            
            # *training_trial_2_key_mapping* updates
            
            # if training_trial_2_key_mapping is starting this frame...
            if training_trial_2_key_mapping.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                training_trial_2_key_mapping.frameNStart = frameN  # exact frame index
                training_trial_2_key_mapping.tStart = t  # local t and not account for scr refresh
                training_trial_2_key_mapping.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_trial_2_key_mapping, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_trial_2_key_mapping.started')
                # update status
                training_trial_2_key_mapping.status = STARTED
                training_trial_2_key_mapping.setAutoDraw(True)
            
            # if training_trial_2_key_mapping is active this frame...
            if training_trial_2_key_mapping.status == STARTED:
                # update params
                pass
            
            # *training_trial_2_key* updates
            waitOnFlip = False
            
            # if training_trial_2_key is starting this frame...
            if training_trial_2_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                training_trial_2_key.frameNStart = frameN  # exact frame index
                training_trial_2_key.tStart = t  # local t and not account for scr refresh
                training_trial_2_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_trial_2_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_trial_2_key.started')
                # update status
                training_trial_2_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(training_trial_2_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(training_trial_2_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if training_trial_2_key.status == STARTED and not waitOnFlip:
                theseKeys = training_trial_2_key.getKeys(keyList=['d', 'f', 'v', 'space', 'n', 'j', 'k'], ignoreKeys=["escape"], waitRelease=False)
                _training_trial_2_key_allKeys.extend(theseKeys)
                if len(_training_trial_2_key_allKeys):
                    training_trial_2_key.keys = _training_trial_2_key_allKeys[-1].name  # just the last key pressed
                    training_trial_2_key.rt = _training_trial_2_key_allKeys[-1].rt
                    training_trial_2_key.duration = _training_trial_2_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in training_trial_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "training_trial_2" ---
        for thisComponent in training_trial_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('training_trial_2.stopped', globalClock.getTime(format='float'))
        # check responses
        if training_trial_2_key.keys in ['', [], None]:  # No response was made
            training_trial_2_key.keys = None
        training_loop_2.addData('training_trial_2_key.keys',training_trial_2_key.keys)
        if training_trial_2_key.keys != None:  # we had a response
            training_loop_2.addData('training_trial_2_key.rt', training_trial_2_key.rt)
            training_loop_2.addData('training_trial_2_key.duration', training_trial_2_key.duration)
        # the Routine "training_trial_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'training_loop_2'
    
    
    # --- Prepare to start Routine "instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions.started', globalClock.getTime(format='float'))
    instructions_key.keys = []
    instructions_key.rt = []
    _instructions_key_allKeys = []
    # keep track of which components have finished
    instructionsComponents = [instructions_text, instructions_key]
    for thisComponent in instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_text* updates
        
        # if instructions_text is starting this frame...
        if instructions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_text.frameNStart = frameN  # exact frame index
            instructions_text.tStart = t  # local t and not account for scr refresh
            instructions_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_text.started')
            # update status
            instructions_text.status = STARTED
            instructions_text.setAutoDraw(True)
        
        # if instructions_text is active this frame...
        if instructions_text.status == STARTED:
            # update params
            pass
        
        # *instructions_key* updates
        waitOnFlip = False
        
        # if instructions_key is starting this frame...
        if instructions_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_key.frameNStart = frameN  # exact frame index
            instructions_key.tStart = t  # local t and not account for scr refresh
            instructions_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_key.started')
            # update status
            instructions_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instructions_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instructions_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instructions_key.status == STARTED and not waitOnFlip:
            theseKeys = instructions_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instructions_key_allKeys.extend(theseKeys)
            if len(_instructions_key_allKeys):
                instructions_key.keys = _instructions_key_allKeys[-1].name  # just the last key pressed
                instructions_key.rt = _instructions_key_allKeys[-1].rt
                instructions_key.duration = _instructions_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions.stopped', globalClock.getTime(format='float'))
    # check responses
    if instructions_key.keys in ['', [], None]:  # No response was made
        instructions_key.keys = None
    thisExp.addData('instructions_key.keys',instructions_key.keys)
    if instructions_key.keys != None:  # we had a response
        thisExp.addData('instructions_key.rt', instructions_key.rt)
        thisExp.addData('instructions_key.duration', instructions_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_message" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('practice_message.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    practice_messageComponents = [practice_message_text]
    for thisComponent in practice_messageComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "practice_message" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practice_message_text* updates
        
        # if practice_message_text is starting this frame...
        if practice_message_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_message_text.frameNStart = frameN  # exact frame index
            practice_message_text.tStart = t  # local t and not account for scr refresh
            practice_message_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_message_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_message_text.started')
            # update status
            practice_message_text.status = STARTED
            practice_message_text.setAutoDraw(True)
        
        # if practice_message_text is active this frame...
        if practice_message_text.status == STARTED:
            # update params
            pass
        
        # if practice_message_text is stopping this frame...
        if practice_message_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > practice_message_text.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                practice_message_text.tStop = t  # not accounting for scr refresh
                practice_message_text.tStopRefresh = tThisFlipGlobal  # on global time
                practice_message_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_message_text.stopped')
                # update status
                practice_message_text.status = FINISHED
                practice_message_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice_messageComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_message" ---
    for thisComponent in practice_messageComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('practice_message.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "practice_first_step" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('practice_first_step.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from practice_first_step_code
    # Message that displays for invalid keypresses, blank to begin with
    keypressMsg = ""
    
    event.clearEvents()
    practice_first_step_valid_keys.keys = []
    practice_first_step_valid_keys.rt = []
    _practice_first_step_valid_keys_allKeys = []
    practice_first_step_invalid_keys.keys = []
    practice_first_step_invalid_keys.rt = []
    _practice_first_step_invalid_keys_allKeys = []
    # keep track of which components have finished
    practice_first_stepComponents = [practice_first_step_text, practice_first_step_image, practice_first_step_key_mapping, practice_first_step_valid_keys, practice_first_step_invalid_keys]
    for thisComponent in practice_first_stepComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "practice_first_step" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from practice_first_step_code
        invalid = event.getKeys(keyList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        
        if len(invalid) != 0:
            keypressMsg = "invalid keypress, please only use the keys indicated"
        
        # *practice_first_step_text* updates
        
        # if practice_first_step_text is starting this frame...
        if practice_first_step_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_first_step_text.frameNStart = frameN  # exact frame index
            practice_first_step_text.tStart = t  # local t and not account for scr refresh
            practice_first_step_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_first_step_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_first_step_text.started')
            # update status
            practice_first_step_text.status = STARTED
            practice_first_step_text.setAutoDraw(True)
        
        # if practice_first_step_text is active this frame...
        if practice_first_step_text.status == STARTED:
            # update params
            practice_first_step_text.setText(keypressMsg, log=False)
        
        # *practice_first_step_image* updates
        
        # if practice_first_step_image is starting this frame...
        if practice_first_step_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_first_step_image.frameNStart = frameN  # exact frame index
            practice_first_step_image.tStart = t  # local t and not account for scr refresh
            practice_first_step_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_first_step_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_first_step_image.started')
            # update status
            practice_first_step_image.status = STARTED
            practice_first_step_image.setAutoDraw(True)
        
        # if practice_first_step_image is active this frame...
        if practice_first_step_image.status == STARTED:
            # update params
            pass
        
        # *practice_first_step_key_mapping* updates
        
        # if practice_first_step_key_mapping is starting this frame...
        if practice_first_step_key_mapping.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_first_step_key_mapping.frameNStart = frameN  # exact frame index
            practice_first_step_key_mapping.tStart = t  # local t and not account for scr refresh
            practice_first_step_key_mapping.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_first_step_key_mapping, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_first_step_key_mapping.started')
            # update status
            practice_first_step_key_mapping.status = STARTED
            practice_first_step_key_mapping.setAutoDraw(True)
        
        # if practice_first_step_key_mapping is active this frame...
        if practice_first_step_key_mapping.status == STARTED:
            # update params
            pass
        
        # *practice_first_step_valid_keys* updates
        waitOnFlip = False
        
        # if practice_first_step_valid_keys is starting this frame...
        if practice_first_step_valid_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_first_step_valid_keys.frameNStart = frameN  # exact frame index
            practice_first_step_valid_keys.tStart = t  # local t and not account for scr refresh
            practice_first_step_valid_keys.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_first_step_valid_keys, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_first_step_valid_keys.started')
            # update status
            practice_first_step_valid_keys.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(practice_first_step_valid_keys.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(practice_first_step_valid_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if practice_first_step_valid_keys.status == STARTED and not waitOnFlip:
            theseKeys = practice_first_step_valid_keys.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _practice_first_step_valid_keys_allKeys.extend(theseKeys)
            if len(_practice_first_step_valid_keys_allKeys):
                practice_first_step_valid_keys.keys = _practice_first_step_valid_keys_allKeys[-1].name  # just the last key pressed
                practice_first_step_valid_keys.rt = _practice_first_step_valid_keys_allKeys[-1].rt
                practice_first_step_valid_keys.duration = _practice_first_step_valid_keys_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *practice_first_step_invalid_keys* updates
        waitOnFlip = False
        
        # if practice_first_step_invalid_keys is starting this frame...
        if practice_first_step_invalid_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_first_step_invalid_keys.frameNStart = frameN  # exact frame index
            practice_first_step_invalid_keys.tStart = t  # local t and not account for scr refresh
            practice_first_step_invalid_keys.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_first_step_invalid_keys, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_first_step_invalid_keys.started')
            # update status
            practice_first_step_invalid_keys.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(practice_first_step_invalid_keys.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(practice_first_step_invalid_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if practice_first_step_invalid_keys.status == STARTED and not waitOnFlip:
            theseKeys = practice_first_step_invalid_keys.getKeys(keyList=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], ignoreKeys=["escape"], waitRelease=False)
            _practice_first_step_invalid_keys_allKeys.extend(theseKeys)
            if len(_practice_first_step_invalid_keys_allKeys):
                practice_first_step_invalid_keys.keys = _practice_first_step_invalid_keys_allKeys[-1].name  # just the last key pressed
                practice_first_step_invalid_keys.rt = _practice_first_step_invalid_keys_allKeys[-1].rt
                practice_first_step_invalid_keys.duration = _practice_first_step_invalid_keys_allKeys[-1].duration
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice_first_stepComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_first_step" ---
    for thisComponent in practice_first_stepComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('practice_first_step.stopped', globalClock.getTime(format='float'))
    # check responses
    if practice_first_step_valid_keys.keys in ['', [], None]:  # No response was made
        practice_first_step_valid_keys.keys = None
    thisExp.addData('practice_first_step_valid_keys.keys',practice_first_step_valid_keys.keys)
    if practice_first_step_valid_keys.keys != None:  # we had a response
        thisExp.addData('practice_first_step_valid_keys.rt', practice_first_step_valid_keys.rt)
        thisExp.addData('practice_first_step_valid_keys.duration', practice_first_step_valid_keys.duration)
    # check responses
    if practice_first_step_invalid_keys.keys in ['', [], None]:  # No response was made
        practice_first_step_invalid_keys.keys = None
    thisExp.addData('practice_first_step_invalid_keys.keys',practice_first_step_invalid_keys.keys)
    if practice_first_step_invalid_keys.keys != None:  # we had a response
        thisExp.addData('practice_first_step_invalid_keys.rt', practice_first_step_invalid_keys.rt)
        thisExp.addData('practice_first_step_invalid_keys.duration', practice_first_step_invalid_keys.duration)
    thisExp.nextEntry()
    # the Routine "practice_first_step" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions/practice.xlsx'),
        seed=None, name='practice_loop')
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    
    for thisPractice_loop in practice_loop:
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # --- Prepare to start Routine "isi" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('isi.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        isiComponents = [isi_text]
        for thisComponent in isiComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "isi" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.1:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *isi_text* updates
            
            # if isi_text is starting this frame...
            if isi_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                isi_text.frameNStart = frameN  # exact frame index
                isi_text.tStart = t  # local t and not account for scr refresh
                isi_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(isi_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'isi_text.started')
                # update status
                isi_text.status = STARTED
                isi_text.setAutoDraw(True)
            
            # if isi_text is active this frame...
            if isi_text.status == STARTED:
                # update params
                pass
            
            # if isi_text is stopping this frame...
            if isi_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > isi_text.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    isi_text.tStop = t  # not accounting for scr refresh
                    isi_text.tStopRefresh = tThisFlipGlobal  # on global time
                    isi_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'isi_text.stopped')
                    # update status
                    isi_text.status = FINISHED
                    isi_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in isiComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "isi" ---
        for thisComponent in isiComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('isi.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.100000)
        
        # --- Prepare to start Routine "practice_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice_trial.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from practice_trial_code
        # Message that displays for invalid keypresses, blank to begin with
        keypressMsg = ""
        
        event.clearEvents()
        practice_trial_image.setImage(practiceStim)
        practice_trial_valid_keys.keys = []
        practice_trial_valid_keys.rt = []
        _practice_trial_valid_keys_allKeys = []
        practice_trial_invalid_keys.keys = []
        practice_trial_invalid_keys.rt = []
        _practice_trial_invalid_keys_allKeys = []
        # keep track of which components have finished
        practice_trialComponents = [practice_trial_text, practice_trial_key_mapping, practice_trial_image, practice_trial_valid_keys, practice_trial_invalid_keys]
        for thisComponent in practice_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from practice_trial_code
            invalid = event.getKeys(keyList = ['a', 'b', 'c', 'e', 'g', 'h', 'i', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z'])
            
            if len(invalid) != 0:
                keypressMsg = "invalid keypress, please only use the keys indicated"
            
            # *practice_trial_text* updates
            
            # if practice_trial_text is starting this frame...
            if practice_trial_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_trial_text.frameNStart = frameN  # exact frame index
                practice_trial_text.tStart = t  # local t and not account for scr refresh
                practice_trial_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_trial_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_trial_text.started')
                # update status
                practice_trial_text.status = STARTED
                practice_trial_text.setAutoDraw(True)
            
            # if practice_trial_text is active this frame...
            if practice_trial_text.status == STARTED:
                # update params
                practice_trial_text.setText(keypressMsg, log=False)
            
            # *practice_trial_key_mapping* updates
            
            # if practice_trial_key_mapping is starting this frame...
            if practice_trial_key_mapping.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_trial_key_mapping.frameNStart = frameN  # exact frame index
                practice_trial_key_mapping.tStart = t  # local t and not account for scr refresh
                practice_trial_key_mapping.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_trial_key_mapping, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_trial_key_mapping.started')
                # update status
                practice_trial_key_mapping.status = STARTED
                practice_trial_key_mapping.setAutoDraw(True)
            
            # if practice_trial_key_mapping is active this frame...
            if practice_trial_key_mapping.status == STARTED:
                # update params
                pass
            
            # *practice_trial_image* updates
            
            # if practice_trial_image is starting this frame...
            if practice_trial_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_trial_image.frameNStart = frameN  # exact frame index
                practice_trial_image.tStart = t  # local t and not account for scr refresh
                practice_trial_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_trial_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_trial_image.started')
                # update status
                practice_trial_image.status = STARTED
                practice_trial_image.setAutoDraw(True)
            
            # if practice_trial_image is active this frame...
            if practice_trial_image.status == STARTED:
                # update params
                pass
            
            # *practice_trial_valid_keys* updates
            waitOnFlip = False
            
            # if practice_trial_valid_keys is starting this frame...
            if practice_trial_valid_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_trial_valid_keys.frameNStart = frameN  # exact frame index
                practice_trial_valid_keys.tStart = t  # local t and not account for scr refresh
                practice_trial_valid_keys.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_trial_valid_keys, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_trial_valid_keys.started')
                # update status
                practice_trial_valid_keys.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(practice_trial_valid_keys.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(practice_trial_valid_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if practice_trial_valid_keys.status == STARTED and not waitOnFlip:
                theseKeys = practice_trial_valid_keys.getKeys(keyList=['d', 'f', 'v', 'space', 'n', 'j', 'k'], ignoreKeys=["escape"], waitRelease=False)
                _practice_trial_valid_keys_allKeys.extend(theseKeys)
                if len(_practice_trial_valid_keys_allKeys):
                    practice_trial_valid_keys.keys = _practice_trial_valid_keys_allKeys[-1].name  # just the last key pressed
                    practice_trial_valid_keys.rt = _practice_trial_valid_keys_allKeys[-1].rt
                    practice_trial_valid_keys.duration = _practice_trial_valid_keys_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *practice_trial_invalid_keys* updates
            waitOnFlip = False
            
            # if practice_trial_invalid_keys is starting this frame...
            if practice_trial_invalid_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_trial_invalid_keys.frameNStart = frameN  # exact frame index
                practice_trial_invalid_keys.tStart = t  # local t and not account for scr refresh
                practice_trial_invalid_keys.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_trial_invalid_keys, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_trial_invalid_keys.started')
                # update status
                practice_trial_invalid_keys.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(practice_trial_invalid_keys.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(practice_trial_invalid_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if practice_trial_invalid_keys.status == STARTED and not waitOnFlip:
                theseKeys = practice_trial_invalid_keys.getKeys(keyList=['a', 'b', 'c', 'e', 'g', 'h', 'i', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z'], ignoreKeys=["escape"], waitRelease=False)
                _practice_trial_invalid_keys_allKeys.extend(theseKeys)
                if len(_practice_trial_invalid_keys_allKeys):
                    practice_trial_invalid_keys.keys = _practice_trial_invalid_keys_allKeys[-1].name  # just the last key pressed
                    practice_trial_invalid_keys.rt = _practice_trial_invalid_keys_allKeys[-1].rt
                    practice_trial_invalid_keys.duration = _practice_trial_invalid_keys_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_trial" ---
        for thisComponent in practice_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice_trial.stopped', globalClock.getTime(format='float'))
        # check responses
        if practice_trial_valid_keys.keys in ['', [], None]:  # No response was made
            practice_trial_valid_keys.keys = None
        practice_loop.addData('practice_trial_valid_keys.keys',practice_trial_valid_keys.keys)
        if practice_trial_valid_keys.keys != None:  # we had a response
            practice_loop.addData('practice_trial_valid_keys.rt', practice_trial_valid_keys.rt)
            practice_loop.addData('practice_trial_valid_keys.duration', practice_trial_valid_keys.duration)
        # check responses
        if practice_trial_invalid_keys.keys in ['', [], None]:  # No response was made
            practice_trial_invalid_keys.keys = None
        practice_loop.addData('practice_trial_invalid_keys.keys',practice_trial_invalid_keys.keys)
        if practice_trial_invalid_keys.keys != None:  # we had a response
            practice_loop.addData('practice_trial_invalid_keys.rt', practice_trial_invalid_keys.rt)
            practice_loop.addData('practice_trial_invalid_keys.duration', practice_trial_invalid_keys.duration)
        # the Routine "practice_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'practice_loop'
    
    
    # --- Prepare to start Routine "main_message" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('main_message.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    main_messageComponents = [main_message_text]
    for thisComponent in main_messageComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "main_message" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *main_message_text* updates
        
        # if main_message_text is starting this frame...
        if main_message_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            main_message_text.frameNStart = frameN  # exact frame index
            main_message_text.tStart = t  # local t and not account for scr refresh
            main_message_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(main_message_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'main_message_text.started')
            # update status
            main_message_text.status = STARTED
            main_message_text.setAutoDraw(True)
        
        # if main_message_text is active this frame...
        if main_message_text.status == STARTED:
            # update params
            pass
        
        # if main_message_text is stopping this frame...
        if main_message_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > main_message_text.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                main_message_text.tStop = t  # not accounting for scr refresh
                main_message_text.tStopRefresh = tThisFlipGlobal  # on global time
                main_message_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'main_message_text.stopped')
                # update status
                main_message_text.status = FINISHED
                main_message_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in main_messageComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "main_message" ---
    for thisComponent in main_messageComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('main_message.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    identity_loop = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions/identity.xlsx'),
        seed=None, name='identity_loop')
    thisExp.addLoop(identity_loop)  # add the loop to the experiment
    thisIdentity_loop = identity_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIdentity_loop.rgb)
    if thisIdentity_loop != None:
        for paramName in thisIdentity_loop:
            globals()[paramName] = thisIdentity_loop[paramName]
    
    for thisIdentity_loop in identity_loop:
        currentLoop = identity_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisIdentity_loop.rgb)
        if thisIdentity_loop != None:
            for paramName in thisIdentity_loop:
                globals()[paramName] = thisIdentity_loop[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        fixationComponents = [fixation_text]
        for thisComponent in fixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_text* updates
            
            # if fixation_text is starting this frame...
            if fixation_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_text.frameNStart = frameN  # exact frame index
                fixation_text.tStart = t  # local t and not account for scr refresh
                fixation_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_text.started')
                # update status
                fixation_text.status = STARTED
                fixation_text.setAutoDraw(True)
            
            # if fixation_text is active this frame...
            if fixation_text.status == STARTED:
                # update params
                pass
            
            # if fixation_text is stopping this frame...
            if fixation_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_text.tStartRefresh + fixationDur-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_text.tStop = t  # not accounting for scr refresh
                    fixation_text.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_text.stopped')
                    # update status
                    fixation_text.status = FINISHED
                    fixation_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation.stopped', globalClock.getTime(format='float'))
        # the Routine "fixation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        morph_loop = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('conditions/morph.xlsx'),
            seed=None, name='morph_loop')
        thisExp.addLoop(morph_loop)  # add the loop to the experiment
        thisMorph_loop = morph_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMorph_loop.rgb)
        if thisMorph_loop != None:
            for paramName in thisMorph_loop:
                globals()[paramName] = thisMorph_loop[paramName]
        
        for thisMorph_loop in morph_loop:
            currentLoop = morph_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisMorph_loop.rgb)
            if thisMorph_loop != None:
                for paramName in thisMorph_loop:
                    globals()[paramName] = thisMorph_loop[paramName]
            
            # --- Prepare to start Routine "is_first_step" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('is_first_step.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from is_first_step_code
            if first_step == 1:
                firstStepRep = 1
            else:
                firstStepRep = 0
            # keep track of which components have finished
            is_first_stepComponents = []
            for thisComponent in is_first_stepComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "is_first_step" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in is_first_stepComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "is_first_step" ---
            for thisComponent in is_first_stepComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('is_first_step.stopped', globalClock.getTime(format='float'))
            # the Routine "is_first_step" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            first_step_loop = data.TrialHandler(nReps=firstStepRep, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='first_step_loop')
            thisExp.addLoop(first_step_loop)  # add the loop to the experiment
            thisFirst_step_loop = first_step_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisFirst_step_loop.rgb)
            if thisFirst_step_loop != None:
                for paramName in thisFirst_step_loop:
                    globals()[paramName] = thisFirst_step_loop[paramName]
            
            for thisFirst_step_loop in first_step_loop:
                currentLoop = first_step_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisFirst_step_loop.rgb)
                if thisFirst_step_loop != None:
                    for paramName in thisFirst_step_loop:
                        globals()[paramName] = thisFirst_step_loop[paramName]
                
                # --- Prepare to start Routine "isi" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('isi.started', globalClock.getTime(format='float'))
                # keep track of which components have finished
                isiComponents = [isi_text]
                for thisComponent in isiComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "isi" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.1:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *isi_text* updates
                    
                    # if isi_text is starting this frame...
                    if isi_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        isi_text.frameNStart = frameN  # exact frame index
                        isi_text.tStart = t  # local t and not account for scr refresh
                        isi_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(isi_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'isi_text.started')
                        # update status
                        isi_text.status = STARTED
                        isi_text.setAutoDraw(True)
                    
                    # if isi_text is active this frame...
                    if isi_text.status == STARTED:
                        # update params
                        pass
                    
                    # if isi_text is stopping this frame...
                    if isi_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > isi_text.tStartRefresh + 0.1-frameTolerance:
                            # keep track of stop time/frame for later
                            isi_text.tStop = t  # not accounting for scr refresh
                            isi_text.tStopRefresh = tThisFlipGlobal  # on global time
                            isi_text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'isi_text.stopped')
                            # update status
                            isi_text.status = FINISHED
                            isi_text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in isiComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "isi" ---
                for thisComponent in isiComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('isi.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.100000)
                
                # --- Prepare to start Routine "first_morph_step" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('first_morph_step.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from first_morph_step_code
                # Concatenate variables to select correct image from correct folder
                imageStim = folder + race + gender + emotion + append + ".jpg"
                
                # Message that displays for invalid keypresses, blank to begin with
                keypressMsg = ""
                
                event.clearEvents()
                first_morph_step_image.setImage(imageStim)
                first_morph_step_valid_keys.keys = []
                first_morph_step_valid_keys.rt = []
                _first_morph_step_valid_keys_allKeys = []
                first_morph_step_invalid_keys.keys = []
                first_morph_step_invalid_keys.rt = []
                _first_morph_step_invalid_keys_allKeys = []
                # keep track of which components have finished
                first_morph_stepComponents = [first_morph_step_text, first_morph_step_image, first_morph_step_key_mapping, first_morph_step_valid_keys, first_morph_step_invalid_keys]
                for thisComponent in first_morph_stepComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "first_morph_step" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from first_morph_step_code
                    invalid = event.getKeys(keyList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
                    
                    if len(invalid) != 0:
                        keypressMsg = "invalid keypress, please only use the keys indicated"
                    
                    # *first_morph_step_text* updates
                    
                    # if first_morph_step_text is starting this frame...
                    if first_morph_step_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        first_morph_step_text.frameNStart = frameN  # exact frame index
                        first_morph_step_text.tStart = t  # local t and not account for scr refresh
                        first_morph_step_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(first_morph_step_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'first_morph_step_text.started')
                        # update status
                        first_morph_step_text.status = STARTED
                        first_morph_step_text.setAutoDraw(True)
                    
                    # if first_morph_step_text is active this frame...
                    if first_morph_step_text.status == STARTED:
                        # update params
                        first_morph_step_text.setText(keypressMsg, log=False)
                    
                    # *first_morph_step_image* updates
                    
                    # if first_morph_step_image is starting this frame...
                    if first_morph_step_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        first_morph_step_image.frameNStart = frameN  # exact frame index
                        first_morph_step_image.tStart = t  # local t and not account for scr refresh
                        first_morph_step_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(first_morph_step_image, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'first_morph_step_image.started')
                        # update status
                        first_morph_step_image.status = STARTED
                        first_morph_step_image.setAutoDraw(True)
                    
                    # if first_morph_step_image is active this frame...
                    if first_morph_step_image.status == STARTED:
                        # update params
                        pass
                    
                    # *first_morph_step_key_mapping* updates
                    
                    # if first_morph_step_key_mapping is starting this frame...
                    if first_morph_step_key_mapping.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        first_morph_step_key_mapping.frameNStart = frameN  # exact frame index
                        first_morph_step_key_mapping.tStart = t  # local t and not account for scr refresh
                        first_morph_step_key_mapping.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(first_morph_step_key_mapping, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'first_morph_step_key_mapping.started')
                        # update status
                        first_morph_step_key_mapping.status = STARTED
                        first_morph_step_key_mapping.setAutoDraw(True)
                    
                    # if first_morph_step_key_mapping is active this frame...
                    if first_morph_step_key_mapping.status == STARTED:
                        # update params
                        pass
                    
                    # *first_morph_step_valid_keys* updates
                    waitOnFlip = False
                    
                    # if first_morph_step_valid_keys is starting this frame...
                    if first_morph_step_valid_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        first_morph_step_valid_keys.frameNStart = frameN  # exact frame index
                        first_morph_step_valid_keys.tStart = t  # local t and not account for scr refresh
                        first_morph_step_valid_keys.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(first_morph_step_valid_keys, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'first_morph_step_valid_keys.started')
                        # update status
                        first_morph_step_valid_keys.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(first_morph_step_valid_keys.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(first_morph_step_valid_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if first_morph_step_valid_keys.status == STARTED and not waitOnFlip:
                        theseKeys = first_morph_step_valid_keys.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                        _first_morph_step_valid_keys_allKeys.extend(theseKeys)
                        if len(_first_morph_step_valid_keys_allKeys):
                            first_morph_step_valid_keys.keys = _first_morph_step_valid_keys_allKeys[-1].name  # just the last key pressed
                            first_morph_step_valid_keys.rt = _first_morph_step_valid_keys_allKeys[-1].rt
                            first_morph_step_valid_keys.duration = _first_morph_step_valid_keys_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *first_morph_step_invalid_keys* updates
                    waitOnFlip = False
                    
                    # if first_morph_step_invalid_keys is starting this frame...
                    if first_morph_step_invalid_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        first_morph_step_invalid_keys.frameNStart = frameN  # exact frame index
                        first_morph_step_invalid_keys.tStart = t  # local t and not account for scr refresh
                        first_morph_step_invalid_keys.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(first_morph_step_invalid_keys, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'first_morph_step_invalid_keys.started')
                        # update status
                        first_morph_step_invalid_keys.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(first_morph_step_invalid_keys.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(first_morph_step_invalid_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if first_morph_step_invalid_keys.status == STARTED and not waitOnFlip:
                        theseKeys = first_morph_step_invalid_keys.getKeys(keyList=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], ignoreKeys=["escape"], waitRelease=False)
                        _first_morph_step_invalid_keys_allKeys.extend(theseKeys)
                        if len(_first_morph_step_invalid_keys_allKeys):
                            first_morph_step_invalid_keys.keys = _first_morph_step_invalid_keys_allKeys[-1].name  # just the last key pressed
                            first_morph_step_invalid_keys.rt = _first_morph_step_invalid_keys_allKeys[-1].rt
                            first_morph_step_invalid_keys.duration = _first_morph_step_invalid_keys_allKeys[-1].duration
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in first_morph_stepComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "first_morph_step" ---
                for thisComponent in first_morph_stepComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('first_morph_step.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from first_morph_step_code
                response = first_morph_step_valid_keys.keys
                rt = first_morph_step_valid_keys.rt
                # check responses
                if first_morph_step_valid_keys.keys in ['', [], None]:  # No response was made
                    first_morph_step_valid_keys.keys = None
                first_step_loop.addData('first_morph_step_valid_keys.keys',first_morph_step_valid_keys.keys)
                if first_morph_step_valid_keys.keys != None:  # we had a response
                    first_step_loop.addData('first_morph_step_valid_keys.rt', first_morph_step_valid_keys.rt)
                    first_step_loop.addData('first_morph_step_valid_keys.duration', first_morph_step_valid_keys.duration)
                # check responses
                if first_morph_step_invalid_keys.keys in ['', [], None]:  # No response was made
                    first_morph_step_invalid_keys.keys = None
                first_step_loop.addData('first_morph_step_invalid_keys.keys',first_morph_step_invalid_keys.keys)
                if first_morph_step_invalid_keys.keys != None:  # we had a response
                    first_step_loop.addData('first_morph_step_invalid_keys.rt', first_morph_step_invalid_keys.rt)
                    first_step_loop.addData('first_morph_step_invalid_keys.duration', first_morph_step_invalid_keys.duration)
                # the Routine "first_morph_step" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "add_data" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('add_data.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from add_data_code
                thisExp.addData("image", imageStim)
                thisExp.addData("response", response)
                thisExp.addData("rt", rt)
                # keep track of which components have finished
                add_dataComponents = []
                for thisComponent in add_dataComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "add_data" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in add_dataComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "add_data" ---
                for thisComponent in add_dataComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('add_data.stopped', globalClock.getTime(format='float'))
                # the Routine "add_data" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed firstStepRep repeats of 'first_step_loop'
            
            
            # --- Prepare to start Routine "isnt_first_step" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('isnt_first_step.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from isnt_first_step_loop
            if first_step == 1:
                remainingStepsRep = 0
            else:
                remainingStepsRep = 1
            # keep track of which components have finished
            isnt_first_stepComponents = []
            for thisComponent in isnt_first_stepComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "isnt_first_step" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in isnt_first_stepComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "isnt_first_step" ---
            for thisComponent in isnt_first_stepComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('isnt_first_step.stopped', globalClock.getTime(format='float'))
            # the Routine "isnt_first_step" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            remaining_steps_loop = data.TrialHandler(nReps=remainingStepsRep, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='remaining_steps_loop')
            thisExp.addLoop(remaining_steps_loop)  # add the loop to the experiment
            thisRemaining_steps_loop = remaining_steps_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisRemaining_steps_loop.rgb)
            if thisRemaining_steps_loop != None:
                for paramName in thisRemaining_steps_loop:
                    globals()[paramName] = thisRemaining_steps_loop[paramName]
            
            for thisRemaining_steps_loop in remaining_steps_loop:
                currentLoop = remaining_steps_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisRemaining_steps_loop.rgb)
                if thisRemaining_steps_loop != None:
                    for paramName in thisRemaining_steps_loop:
                        globals()[paramName] = thisRemaining_steps_loop[paramName]
                
                # --- Prepare to start Routine "isi" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('isi.started', globalClock.getTime(format='float'))
                # keep track of which components have finished
                isiComponents = [isi_text]
                for thisComponent in isiComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "isi" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.1:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *isi_text* updates
                    
                    # if isi_text is starting this frame...
                    if isi_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        isi_text.frameNStart = frameN  # exact frame index
                        isi_text.tStart = t  # local t and not account for scr refresh
                        isi_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(isi_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'isi_text.started')
                        # update status
                        isi_text.status = STARTED
                        isi_text.setAutoDraw(True)
                    
                    # if isi_text is active this frame...
                    if isi_text.status == STARTED:
                        # update params
                        pass
                    
                    # if isi_text is stopping this frame...
                    if isi_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > isi_text.tStartRefresh + 0.1-frameTolerance:
                            # keep track of stop time/frame for later
                            isi_text.tStop = t  # not accounting for scr refresh
                            isi_text.tStopRefresh = tThisFlipGlobal  # on global time
                            isi_text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'isi_text.stopped')
                            # update status
                            isi_text.status = FINISHED
                            isi_text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in isiComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "isi" ---
                for thisComponent in isiComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('isi.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.100000)
                
                # --- Prepare to start Routine "rating" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('rating.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from rating_code
                # Concatenate variables to select correct image from correct folder
                imageStim = folder + race + gender + emotion + append + ".jpg"
                
                # Message that displays for invalid keypresses, blank to begin with
                keypressMsg = ""
                
                event.clearEvents()
                rating_image.setImage(imageStim)
                rating_valid_keys.keys = []
                rating_valid_keys.rt = []
                _rating_valid_keys_allKeys = []
                rating_invalid_keys.keys = []
                rating_invalid_keys.rt = []
                _rating_invalid_keys_allKeys = []
                # keep track of which components have finished
                ratingComponents = [rating_text, rating_image, rating_key_mapping, rating_valid_keys, rating_invalid_keys]
                for thisComponent in ratingComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "rating" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from rating_code
                    invalid = event.getKeys(keyList = ['a', 'b', 'c', 'e', 'g', 'h', 'i', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z'])
                    
                    if len(invalid) != 0:
                        keypressMsg = "invalid keypress, please only use the keys indicated"
                    
                    # *rating_text* updates
                    
                    # if rating_text is starting this frame...
                    if rating_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        rating_text.frameNStart = frameN  # exact frame index
                        rating_text.tStart = t  # local t and not account for scr refresh
                        rating_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(rating_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rating_text.started')
                        # update status
                        rating_text.status = STARTED
                        rating_text.setAutoDraw(True)
                    
                    # if rating_text is active this frame...
                    if rating_text.status == STARTED:
                        # update params
                        rating_text.setText(keypressMsg, log=False)
                    
                    # *rating_image* updates
                    
                    # if rating_image is starting this frame...
                    if rating_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        rating_image.frameNStart = frameN  # exact frame index
                        rating_image.tStart = t  # local t and not account for scr refresh
                        rating_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(rating_image, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rating_image.started')
                        # update status
                        rating_image.status = STARTED
                        rating_image.setAutoDraw(True)
                    
                    # if rating_image is active this frame...
                    if rating_image.status == STARTED:
                        # update params
                        pass
                    
                    # *rating_key_mapping* updates
                    
                    # if rating_key_mapping is starting this frame...
                    if rating_key_mapping.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        rating_key_mapping.frameNStart = frameN  # exact frame index
                        rating_key_mapping.tStart = t  # local t and not account for scr refresh
                        rating_key_mapping.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(rating_key_mapping, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rating_key_mapping.started')
                        # update status
                        rating_key_mapping.status = STARTED
                        rating_key_mapping.setAutoDraw(True)
                    
                    # if rating_key_mapping is active this frame...
                    if rating_key_mapping.status == STARTED:
                        # update params
                        pass
                    
                    # *rating_valid_keys* updates
                    waitOnFlip = False
                    
                    # if rating_valid_keys is starting this frame...
                    if rating_valid_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        rating_valid_keys.frameNStart = frameN  # exact frame index
                        rating_valid_keys.tStart = t  # local t and not account for scr refresh
                        rating_valid_keys.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(rating_valid_keys, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rating_valid_keys.started')
                        # update status
                        rating_valid_keys.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(rating_valid_keys.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(rating_valid_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if rating_valid_keys.status == STARTED and not waitOnFlip:
                        theseKeys = rating_valid_keys.getKeys(keyList=['d', 'f', 'v', 'space', 'n', 'j', 'k'], ignoreKeys=["escape"], waitRelease=False)
                        _rating_valid_keys_allKeys.extend(theseKeys)
                        if len(_rating_valid_keys_allKeys):
                            rating_valid_keys.keys = _rating_valid_keys_allKeys[-1].name  # just the last key pressed
                            rating_valid_keys.rt = _rating_valid_keys_allKeys[-1].rt
                            rating_valid_keys.duration = _rating_valid_keys_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *rating_invalid_keys* updates
                    waitOnFlip = False
                    
                    # if rating_invalid_keys is starting this frame...
                    if rating_invalid_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        rating_invalid_keys.frameNStart = frameN  # exact frame index
                        rating_invalid_keys.tStart = t  # local t and not account for scr refresh
                        rating_invalid_keys.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(rating_invalid_keys, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rating_invalid_keys.started')
                        # update status
                        rating_invalid_keys.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(rating_invalid_keys.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(rating_invalid_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if rating_invalid_keys.status == STARTED and not waitOnFlip:
                        theseKeys = rating_invalid_keys.getKeys(keyList=['a', 'b', 'c', 'e', 'g', 'h', 'i', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z'], ignoreKeys=["escape"], waitRelease=False)
                        _rating_invalid_keys_allKeys.extend(theseKeys)
                        if len(_rating_invalid_keys_allKeys):
                            rating_invalid_keys.keys = [key.name for key in _rating_invalid_keys_allKeys]  # storing all keys
                            rating_invalid_keys.rt = [key.rt for key in _rating_invalid_keys_allKeys]
                            rating_invalid_keys.duration = [key.duration for key in _rating_invalid_keys_allKeys]
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in ratingComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "rating" ---
                for thisComponent in ratingComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('rating.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from rating_code
                response = rating_valid_keys.keys
                rt = rating_valid_keys.rt
                # check responses
                if rating_valid_keys.keys in ['', [], None]:  # No response was made
                    rating_valid_keys.keys = None
                remaining_steps_loop.addData('rating_valid_keys.keys',rating_valid_keys.keys)
                if rating_valid_keys.keys != None:  # we had a response
                    remaining_steps_loop.addData('rating_valid_keys.rt', rating_valid_keys.rt)
                    remaining_steps_loop.addData('rating_valid_keys.duration', rating_valid_keys.duration)
                # check responses
                if rating_invalid_keys.keys in ['', [], None]:  # No response was made
                    rating_invalid_keys.keys = None
                remaining_steps_loop.addData('rating_invalid_keys.keys',rating_invalid_keys.keys)
                if rating_invalid_keys.keys != None:  # we had a response
                    remaining_steps_loop.addData('rating_invalid_keys.rt', rating_invalid_keys.rt)
                    remaining_steps_loop.addData('rating_invalid_keys.duration', rating_invalid_keys.duration)
                # the Routine "rating" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "add_data" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('add_data.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from add_data_code
                thisExp.addData("image", imageStim)
                thisExp.addData("response", response)
                thisExp.addData("rt", rt)
                # keep track of which components have finished
                add_dataComponents = []
                for thisComponent in add_dataComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "add_data" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in add_dataComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "add_data" ---
                for thisComponent in add_dataComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('add_data.stopped', globalClock.getTime(format='float'))
                # the Routine "add_data" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed remainingStepsRep repeats of 'remaining_steps_loop'
            
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'morph_loop'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'identity_loop'
    
    
    # --- Prepare to start Routine "end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end.started', globalClock.getTime(format='float'))
    end_key.keys = []
    end_key.rt = []
    _end_key_allKeys = []
    # keep track of which components have finished
    endComponents = [end_text, end_key]
    for thisComponent in endComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.started')
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
        # *end_key* updates
        waitOnFlip = False
        
        # if end_key is starting this frame...
        if end_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_key.frameNStart = frameN  # exact frame index
            end_key.tStart = t  # local t and not account for scr refresh
            end_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_key.started')
            # update status
            end_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_key.status == STARTED and not waitOnFlip:
            theseKeys = end_key.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _end_key_allKeys.extend(theseKeys)
            if len(_end_key_allKeys):
                end_key.keys = _end_key_allKeys[-1].name  # just the last key pressed
                end_key.rt = _end_key_allKeys[-1].rt
                end_key.duration = _end_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end.stopped', globalClock.getTime(format='float'))
    # check responses
    if end_key.keys in ['', [], None]:  # No response was made
        end_key.keys = None
    thisExp.addData('end_key.keys',end_key.keys)
    if end_key.keys != None:  # we had a response
        thisExp.addData('end_key.rt', end_key.rt)
        thisExp.addData('end_key.duration', end_key.duration)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
